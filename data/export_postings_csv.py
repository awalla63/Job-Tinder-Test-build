"""Export the trimmed posting corpus as a CSV, for manual import via the Supabase
table editor instead of `load_postings.py`'s direct Postgres write.

Reuses the exact same filter chain as load_postings.py (dedupe, experience-level
requirement, per-company cap) so the two paths can never silently diverge -- the numbers
in the README are reproducible either way.

Columns match the insertable (non-generated) columns of `public.jobs` exactly, in the
order Supabase's table editor expects for a CSV import: `source` is a literal
'linkedin_seed' on every row, matching the job_source enum. `id`, `employer_id`, and
`created_at` are left for Postgres to default. `embedding` is deliberately omitted --
a 384-float column would roughly triple the file size for no benefit, since the table
editor can't populate a halfvec from a CSV cell anyway. Run `ml/embed.py --target jobs`
after import; it backfills anything with a NULL embedding and is idempotent.

Usage:
    python data/export_postings_csv.py --csv path/to/postings.csv.zip --out jobs_trimmed.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.load_postings import select_rows  # noqa: E402

COLUMNS = [
    "source", "external_id", "company_name", "title", "description", "location",
    "state", "normalized_salary", "pay_period", "formatted_work_type",
    "formatted_experience_level", "remote_allowed", "application_url",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=os.environ.get("POSTINGS_CSV", "data/raw/postings.csv"))
    parser.add_argument("--out", default="data/raw/jobs_trimmed.csv")
    parser.add_argument("--company-cap", type=int, default=10)
    parser.add_argument("--desc-chars", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=0,
                        help="Hard row cap after filtering; 0 = no limit.")
    parser.add_argument("--chunk-rows", type=int, default=0,
                        help="Split output into multiple files of this many rows each "
                             "(e.g. jobs_trimmed.part01.csv, .part02.csv, ...). A "
                             "browser-based table-editor upload of a 150MB+ single file "
                             "is unreliable; chunking into ~10-20MB files is the "
                             "practical path. Rows have no foreign-key dependency on "
                             "each other, so import order across chunks doesn't matter.")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        sys.exit(f"Not found: {path}\nSet POSTINGS_CSV in .env or pass --csv.")

    rows = select_rows(path, args.company_cap, args.limit, args.desc_chars)
    for row in rows:
        row["source"] = "linkedin_seed"
        # Postgres' CSV importer reads an empty field as NULL for numeric/boolean
        # columns, but "False" (Python's str(bool)) is not valid boolean input --
        # it wants lowercase true/false.
        row["remote_allowed"] = "true" if row["remote_allowed"] else "false"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def write_csv(target: Path, subset: list[dict]) -> None:
        with target.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(subset)

    written: list[Path] = []
    if args.chunk_rows:
        stem, suffix = out_path.stem, out_path.suffix
        for i in range(0, len(rows), args.chunk_rows):
            part = (i // args.chunk_rows) + 1
            target = out_path.with_name(f"{stem}.part{part:02d}{suffix}")
            write_csv(target, rows[i:i + args.chunk_rows])
            written.append(target)
    else:
        write_csv(out_path, rows)
        written.append(out_path)

    print(f"Wrote {len(rows):,} rows across {len(written)} file(s):")
    for target in written:
        print(f"  {target}  ({target.stat().st_size / 1e6:.1f} MB)")
    print("\nIn the Supabase dashboard: Table Editor -> jobs -> Insert -> Import data from CSV.")
    if len(written) > 1:
        print("Import each part separately -- rows have no foreign-key dependency on")
        print("each other, so the order you import them in doesn't matter.")
    print("After import, run:  python ml/embed.py --target jobs")


if __name__ == "__main__":
    main()
