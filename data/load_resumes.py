"""Load Resume.csv into `seeker_profiles` as demo candidates.

This exists to solve employer-side cold start. Without it, the first employer to sign up
posts a job and sees an empty candidate deck, which makes the product look broken before
it has had a chance to work.

These 2,484 resumes are also a genuinely held-out evaluation set: we verified zero
normalized overlap between them and the 643 resumes in the HF fit dataset the model
trains on, so ranking them is a fair test rather than a memorisation check.

Demo rows carry is_demo = true, have no auth.users row, and are excluded from match
creation -- an employer can swipe on them, but no mutual match will ever fire.

Usage:
    python data/load_resumes.py --csv C:/Users/andre/Downloads/Resume.csv.zip
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.db import connect  # noqa: E402
from data.load_postings import open_csv  # noqa: E402

csv.field_size_limit(10**9)

# "Jan 2013 to Current", "2015 - 2019", "06/2012 to 08/2014"
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")


def estimate_years_experience(text: str) -> float | None:
    """Crude span between the earliest and latest year mentioned.

    Deliberately crude: it feeds a single LightGBM feature that the model is free to
    ignore, and over-engineering resume date parsing is a rabbit hole. Capped at 45 to
    keep graduation years and stray dates from producing nonsense.
    """
    years = [int(y) for y in YEAR_RE.findall(text or "")]
    if len(years) < 2:
        return None
    span = max(years) - min(years)
    return float(span) if 0 < span <= 45 else None


def headline_from(text: str) -> str:
    """First meaningful line, which in this corpus is the job title."""
    for line in (text or "").splitlines():
        line = line.strip()
        if len(line) > 3:
            return line[:120]
    return (text or "")[:120].strip()


INSERT_SQL = """
insert into public.seeker_profiles (
    is_demo, display_name, headline, resume_text, years_experience, predicted_category
) values (true, %(display_name)s, %(headline)s, %(resume_text)s, %(years_experience)s, %(category)s)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=os.environ.get("RESUMES_CSV", "data/raw/Resume.csv"))
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--replace", action="store_true",
                        help="Delete existing demo profiles before loading.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        sys.exit(f"Not found: {path}\nSet RESUMES_CSV in .env or pass --csv.")

    rows = []
    with open_csv(path) as fh:
        for row in csv.DictReader(fh):
            text = (row.get("Resume_str") or "").strip()
            if len(text) < 200:
                continue
            category = (row.get("Category") or "").strip()
            rows.append({
                "display_name": f"Candidate {row.get('ID', '')}".strip(),
                "headline": headline_from(text),
                "resume_text": text[:args.max_chars],
                "years_experience": estimate_years_experience(text),
                "category": category or None,
            })

    print(f"{len(rows):,} demo resumes parsed")
    if args.dry_run:
        for r in rows[:3]:
            print(f"  {r['category']:>24} | {r['years_experience']} yrs | {r['headline'][:60]}")
        return

    with connect() as conn, conn.cursor() as cur:
        if args.replace:
            cur.execute("delete from public.seeker_profiles where is_demo")
            print(f"  removed {cur.rowcount:,} existing demo profiles")
        cur.executemany(INSERT_SQL, rows)
        conn.commit()

    print(f"Loaded {len(rows):,} demo profiles. Next: python ml/embed.py --target seekers")


if __name__ == "__main__":
    main()
