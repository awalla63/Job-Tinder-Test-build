"""Load the LinkedIn postings corpus into `jobs`, trimming it on the way in.

The raw file is 123,849 rows / 516 MB uncompressed. The Supabase free tier caps the
database at 500 MB *including indexes*, so loading it whole is not an option -- but most
of what we drop is genuine fat rather than a random sample:

    All rows                                                    123,849
    - missing title / description <700 / Volunteer+Other / closed  118,268
    - duplicate (company, title), keeping the most-viewed copy      92,656
    - require a known experience level                              70,326
    - cap 10 postings per company                                   44,145   (141 MB)

The dedupe step alone removes 26,291 rows (21%): staffing agencies repost the same
listing dozens of times. Leaving them in hurts ranking more than it hurts storage --
one agency's reposts will otherwise flood a deck and make the app look broken.

Usage:
    python data/load_postings.py                      # defaults, ~44k rows
    python data/load_postings.py --company-cap 25     # looser, ~56k rows
    python data/load_postings.py --limit 0 --company-cap 0   # everything (local PG / Pro)
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.db import connect, create_vector_indexes, drop_vector_indexes  # noqa: E402

# Description fields in this file exceed Python's default CSV field cap.
csv.field_size_limit(10**9)

MIN_DESC_CHARS = 700
EXCLUDED_WORK_TYPES = {"Volunteer", "Other"}

# "Charlotte, NC" / "Greater Boston Area" / "North Carolina, United States"
STATE_RE = re.compile(r",\s*([A-Z]{2})\b")

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY", "district of columbia": "DC",
}


def parse_state(location: str) -> str | None:
    if not location:
        return None
    m = STATE_RE.search(location)
    if m:
        return m.group(1)
    low = location.lower()
    for name, code in US_STATES.items():
        if name in low:
            return code
    return None


def as_float(value: str) -> float | None:
    """Tolerant numeric parse.

    `views` and `applies` arrive as "2.0", so int() on them raises -- hence float first.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "t", "yes"}


def open_csv(path: Path):
    """Stream the CSV whether it arrives as .csv or .csv.zip."""
    if path.suffix == ".zip":
        zf = zipfile.ZipFile(path)
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        return io.TextIOWrapper(zf.open(name), encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def select_rows(path: Path, company_cap: int, limit: int, desc_chars: int) -> list[dict]:
    stats = defaultdict(int)
    survivors: list[dict] = []
    best_by_key: dict[tuple[str, str], int] = {}  # (company, title) -> index in survivors

    with open_csv(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stats["total"] += 1

            title = (row.get("title") or "").strip()
            desc = row.get("description") or ""
            if not title or not desc:
                stats["no_title_or_desc"] += 1
                continue
            if len(desc) < MIN_DESC_CHARS:
                stats["short_desc"] += 1
                continue
            if (row.get("formatted_work_type") or "") in EXCLUDED_WORK_TYPES:
                stats["excluded_work_type"] += 1
                continue
            if (row.get("closed_time") or "").strip():
                stats["closed"] += 1
                continue
            if not (row.get("formatted_experience_level") or "").strip():
                stats["no_experience_level"] += 1
                continue

            company = (row.get("company_name") or "").strip()
            views = as_float(row.get("views")) or 0.0
            key = (company.lower(), title.lower())

            record = {
                "external_id": row.get("job_id"),
                "company_name": company,
                "title": title,
                "description": desc[:desc_chars] if desc_chars else desc,
                "location": row.get("location") or None,
                "state": parse_state(row.get("location") or ""),
                "normalized_salary": as_float(row.get("normalized_salary")),
                "pay_period": row.get("pay_period") or None,
                "formatted_work_type": row.get("formatted_work_type") or None,
                "formatted_experience_level": row.get("formatted_experience_level") or None,
                "remote_allowed": as_bool(row.get("remote_allowed")),
                "application_url": row.get("application_url") or row.get("job_posting_url"),
                "_views": views,
            }

            # Dedupe on (company, title), keeping whichever copy has the most views.
            prior = best_by_key.get(key)
            if prior is not None:
                stats["duplicate"] += 1
                if views > survivors[prior]["_views"]:
                    survivors[prior] = record
                continue

            best_by_key[key] = len(survivors)
            survivors.append(record)

    stats["after_dedupe"] = len(survivors)

    # Cap per company, preferring the most-viewed listings.
    if company_cap:
        per_company: dict[str, int] = defaultdict(int)
        capped: list[dict] = []
        for record in sorted(survivors, key=lambda r: -r["_views"]):
            company = record["company_name"].lower()
            if per_company[company] >= company_cap:
                stats["over_company_cap"] += 1
                continue
            per_company[company] += 1
            capped.append(record)
        survivors = capped

    if limit:
        survivors = survivors[:limit]

    stats["kept"] = len(survivors)
    for record in survivors:
        record.pop("_views", None)

    print("Filter chain:")
    for key in ("total", "no_title_or_desc", "short_desc", "excluded_work_type", "closed",
                "no_experience_level", "duplicate", "after_dedupe", "over_company_cap", "kept"):
        if key in stats:
            print(f"  {key:22s} {stats[key]:>8,}")
    return survivors


INSERT_SQL = """
insert into public.jobs (
    source, external_id, company_name, title, description, location, state,
    normalized_salary, pay_period, formatted_work_type, formatted_experience_level,
    remote_allowed, application_url
) values (
    'linkedin_seed', %(external_id)s, %(company_name)s, %(title)s, %(description)s,
    %(location)s, %(state)s, %(normalized_salary)s, %(pay_period)s,
    %(formatted_work_type)s, %(formatted_experience_level)s, %(remote_allowed)s,
    %(application_url)s
)
on conflict (source, external_id) do update set
    title = excluded.title,
    description = excluded.description,
    normalized_salary = excluded.normalized_salary,
    application_url = excluded.application_url
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=os.environ.get("POSTINGS_CSV", "data/raw/postings.csv"))
    parser.add_argument("--company-cap", type=int, default=10,
                        help="Max postings per company; 0 disables the cap.")
    parser.add_argument("--desc-chars", type=int, default=4000,
                        help="Truncate descriptions to this many characters; 0 keeps them whole.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Hard row cap after all filtering; 0 means no limit.")
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the filter chain without touching the database.")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        sys.exit(f"Not found: {path}\nSet POSTINGS_CSV in .env or pass --csv.")

    rows = select_rows(path, args.company_cap, args.limit, args.desc_chars)
    approx_mb = sum(len(r["description"]) for r in rows) / 1e6
    print(f"\n{len(rows):,} postings selected (~{approx_mb:.0f} MB of description text)")

    if args.dry_run:
        print("Dry run; nothing written.")
        return

    with connect() as conn:
        drop_vector_indexes(conn)
        with conn.cursor() as cur:
            for i in range(0, len(rows), args.batch):
                cur.executemany(INSERT_SQL, rows[i:i + args.batch])
                conn.commit()
                print(f"  inserted {min(i + args.batch, len(rows)):,}/{len(rows):,}", end="\r")
        print()
        print("Rebuilding HNSW indexes...")
        create_vector_indexes(conn)

    print("Done. Next: python ml/embed.py --target jobs")


if __name__ == "__main__":
    main()
