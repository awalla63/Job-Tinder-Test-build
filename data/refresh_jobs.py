"""Pull fresh listings from live sources into `jobs`.

Run manually for now; this is the script a daily cron would call later.

    python data/refresh_jobs.py --keywords "data analyst,software engineer" --max-per-keyword 500

New connectors (Greenhouse, Lever, Ashby) slot in by adding an entry to SOURCES: each
needs a callable yielding raw records and a mapper producing the `jobs` column dict.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import usajobs_client  # noqa: E402
from data.db import connect  # noqa: E402

DEFAULT_KEYWORDS = [
    "software engineer", "data analyst", "data scientist", "accountant",
    "human resources", "nurse", "project manager", "cybersecurity",
]

UPSERT_SQL = """
insert into public.jobs (
    source, external_id, company_name, title, description, location, state,
    normalized_salary, pay_period, formatted_work_type, formatted_experience_level,
    remote_allowed, application_url, posted_at
) values (
    %(source)s, %(external_id)s, %(company_name)s, %(title)s, %(description)s,
    %(location)s, %(state)s, %(normalized_salary)s, %(pay_period)s,
    %(formatted_work_type)s, %(formatted_experience_level)s, %(remote_allowed)s,
    %(application_url)s, %(posted_at)s
)
on conflict (source, external_id) do update set
    title = excluded.title,
    description = excluded.description,
    normalized_salary = excluded.normalized_salary,
    application_url = excluded.application_url,
    posted_at = excluded.posted_at,
    -- Text changed, so the stored vector is stale. Null it; ml/embed.py picks up
    -- anything with a null embedding on its next run.
    embedding = case
        when public.jobs.description is distinct from excluded.description then null
        else public.jobs.embedding
    end
"""


def refresh_usajobs(keywords: list[str], max_per_keyword: int, days_posted: int) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    for keyword in keywords:
        print(f"  USAJOBS: {keyword!r} ... ", end="", flush=True)
        count = 0
        for item in usajobs_client.search(
            keyword, days_posted=days_posted, max_results=max_per_keyword
        ):
            mapped = usajobs_client.map_record(item)
            if not mapped or not mapped["external_id"] or mapped["external_id"] in seen:
                continue
            seen.add(mapped["external_id"])
            mapped["source"] = "usajobs"
            records.append(mapped)
            count += 1
        print(f"{count} new")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keywords", default=",".join(DEFAULT_KEYWORDS))
    parser.add_argument("--max-per-keyword", type=int, default=500)
    parser.add_argument("--days-posted", type=int, default=30, help="0-60")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    print(f"Refreshing {len(keywords)} keyword(s) from USAJOBS")

    records = refresh_usajobs(keywords, args.max_per_keyword, args.days_posted)
    print(f"\n{len(records):,} unique listings mapped")

    if args.dry_run:
        for r in records[:3]:
            print(f"  {r['title'][:60]:60} | {r['location']} | {r['application_url']}")
        return
    if not records:
        return

    with connect() as conn, conn.cursor() as cur:
        cur.executemany(UPSERT_SQL, records)
        conn.commit()

    print("Upserted. Next: python ml/embed.py --target jobs   (embeds anything new)")


if __name__ == "__main__":
    main()
