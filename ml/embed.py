"""Fill in missing embeddings for jobs and seeker profiles.

Resumable by design: it only ever selects rows WHERE embedding IS NULL, so an
interrupted run picks up where it stopped, and `refresh_jobs.py` nulling the embedding
of a changed posting is enough to get it re-embedded on the next pass.

    python ml/embed.py --target jobs
    python ml/embed.py --target seekers
    python ml/embed.py --target all

Expect roughly 5-15 minutes for 44k postings on CPU.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.db import connect, create_vector_indexes, drop_vector_indexes, to_pgvector  # noqa: E402
from ml.encoder import get_encoder, job_text  # noqa: E402


def embed_jobs(conn, encoder, batch: int, limit: int | None) -> int:
    total = 0
    started = time.time()
    while True:
        with conn.cursor() as cur:
            cur.execute(
                "select id, title, company_name, description from public.jobs "
                "where embedding is null limit %s",
                (batch,),
            )
            rows = cur.fetchall()
        if not rows:
            break

        ids = [r[0] for r in rows]
        texts = [job_text(r[1], r[2], r[3]) for r in rows]
        vectors = encoder.encode_jobs(texts)

        with conn.cursor() as cur:
            cur.executemany(
                "update public.jobs set embedding = %s::halfvec where id = %s",
                [(to_pgvector(v), i) for v, i in zip(vectors, ids)],
            )
        conn.commit()

        total += len(rows)
        rate = total / max(1e-6, time.time() - started)
        print(f"  jobs: {total:,} embedded ({rate:.0f}/s)", end="\r")
        if limit and total >= limit:
            break
    if total:
        print()
    return total


def embed_seekers(conn, encoder, batch: int, limit: int | None) -> int:
    total = 0
    while True:
        with conn.cursor() as cur:
            cur.execute(
                "select id, resume_text from public.seeker_profiles "
                "where embedding is null and resume_text <> '' limit %s",
                (batch,),
            )
            rows = cur.fetchall()
        if not rows:
            break

        ids = [r[0] for r in rows]
        vectors = encoder.encode_resumes([r[1] for r in rows])

        with conn.cursor() as cur:
            cur.executemany(
                "update public.seeker_profiles set embedding = %s::halfvec where id = %s",
                [(to_pgvector(v), i) for v, i in zip(vectors, ids)],
            )
        conn.commit()

        total += len(rows)
        print(f"  seekers: {total:,} embedded", end="\r")
        if limit and total >= limit:
            break
    if total:
        print()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("jobs", "seekers", "all"), default="all")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Stop after N rows (0 = all).")
    parser.add_argument("--keep-indexes", action="store_true",
                        help="Skip the drop/rebuild of HNSW indexes around the backfill.")
    args = parser.parse_args()

    print("Loading encoder...")
    encoder = get_encoder()

    with connect() as conn:
        # Updating 44k vectors with the HNSW graph live is far slower than rebuilding it
        # once afterwards.
        if not args.keep_indexes:
            drop_vector_indexes(conn)

        limit = args.limit or None
        if args.target in ("jobs", "all"):
            embed_jobs(conn, encoder, args.batch, limit)
        if args.target in ("seekers", "all"):
            embed_seekers(conn, encoder, args.batch, limit)

        if not args.keep_indexes:
            print("Rebuilding HNSW indexes...")
            create_vector_indexes(conn)

    print("Done.")


if __name__ == "__main__":
    main()
