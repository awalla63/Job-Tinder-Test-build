"""Fill in missing embeddings for jobs and seeker profiles.

Resumable by design: it only ever selects rows WHERE embedding IS NULL, so an
interrupted run picks up where it stopped, and `refresh_jobs.py` nulling the embedding
of a changed posting is enough to get it re-embedded on the next pass.

A single long-lived connection is not used across the whole run: pooled Postgres
connections (Supabase's Supavisor pooler in particular) can be closed server-side mid
-job on a run that takes this long, and a plain `with connect() as conn:` wrapping the
entire loop has no way to recover from that. Instead each batch opens its own short
-lived connection; if a batch's connection drops, nothing was committed for it, so
retrying with a fresh connection just re-selects the same still-NULL rows -- no
duplicate work, no lost progress.

    python ml/embed.py --target jobs
    python ml/embed.py --target seekers
    python ml/embed.py --target all

Expect roughly 5-15 minutes for 44k postings on CPU with a stable connection; slower,
and retried in batches, over a flaky one.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psycopg  # noqa: E402

from data.db import connect, create_vector_indexes, drop_vector_indexes, to_pgvector  # noqa: E402
from ml.encoder import get_encoder, job_text  # noqa: E402

MAX_CONSECUTIVE_FAILURES = 15
RETRY_BACKOFF_SECONDS = 3


def _fetch_batch(select_sql: str, batch: int) -> list:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(select_sql, (batch,))
        return cur.fetchall()


def _write_batch(update_sql: str, params: list[tuple]) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(update_sql, params)
        conn.commit()


def _run_resilient(label: str, step) -> None:
    """Call `step()` exactly once, retrying through dropped connections to do it.

    `step` raises psycopg.OperationalError on a connection failure (both _fetch_batch
    and _write_batch do, since each opens its own connection); on success it returns
    immediately, handing control back to the caller's loop so it can re-check whether
    there is more work left (via the `done` flag each caller closes over) rather than
    this function looping internally forever. A consecutive-failure cap stops retrying
    if the database is genuinely unreachable rather than just transiently dropping
    connections.
    """
    failures = 0
    while True:
        try:
            step()
            return
        except psycopg.OperationalError as exc:
            failures += 1
            if failures > MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{label}: {failures} consecutive connection failures, giving up. "
                    f"Last error: {exc}"
                ) from exc
            print(f"\n  {label}: connection dropped ({exc}); "
                  f"reconnecting in {RETRY_BACKOFF_SECONDS}s "
                  f"(attempt {failures}/{MAX_CONSECUTIVE_FAILURES})...")
            time.sleep(RETRY_BACKOFF_SECONDS)


def embed_jobs(encoder, batch: int, limit: int | None) -> int:
    select_sql = (
        "select id, title, company_name, description from public.jobs "
        "where embedding is null limit %s"
    )
    total = 0
    started = time.time()
    done = False

    def step():
        nonlocal total, done
        rows = _fetch_batch(select_sql, batch)
        if not rows:
            done = True
            return

        ids = [r[0] for r in rows]
        texts = [job_text(r[1], r[2], r[3]) for r in rows]
        vectors = encoder.encode_jobs(texts)
        _write_batch(
            "update public.jobs set embedding = %s::halfvec where id = %s",
            [(to_pgvector(v), i) for v, i in zip(vectors, ids)],
        )

        total += len(rows)
        rate = total / max(1e-6, time.time() - started)
        print(f"  jobs: {total:,} embedded ({rate:.1f}/s)", end="\r")
        if limit and total >= limit:
            done = True

    while not done:
        _run_resilient("jobs", step)
    if total:
        print()
    return total


def embed_seekers(encoder, batch: int, limit: int | None) -> int:
    select_sql = (
        "select id, resume_text from public.seeker_profiles "
        "where embedding is null and resume_text <> '' limit %s"
    )
    total = 0
    done = False

    def step():
        nonlocal total, done
        rows = _fetch_batch(select_sql, batch)
        if not rows:
            done = True
            return

        ids = [r[0] for r in rows]
        vectors = encoder.encode_resumes([r[1] for r in rows])
        _write_batch(
            "update public.seeker_profiles set embedding = %s::halfvec where id = %s",
            [(to_pgvector(v), i) for v, i in zip(vectors, ids)],
        )

        total += len(rows)
        print(f"  seekers: {total:,} embedded", end="\r")
        if limit and total >= limit:
            done = True

    while not done:
        _run_resilient("seekers", step)
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

    if not args.keep_indexes:
        with connect() as conn:
            drop_vector_indexes(conn)

    limit = args.limit or None
    try:
        if args.target in ("jobs", "all"):
            embed_jobs(encoder, args.batch, limit)
        if args.target in ("seekers", "all"):
            embed_seekers(encoder, args.batch, limit)
    finally:
        # Rebuilt even on failure/interruption: a partially-embedded table with no
        # index at all is worse than one with an index covering what's done so far,
        # and leaving it dropped after a crash is exactly what happened last run.
        if not args.keep_indexes:
            print("Rebuilding HNSW indexes...")
            with connect() as conn:
                create_vector_indexes(conn)

    print("Done.")


if __name__ == "__main__":
    main()
