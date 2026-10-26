"""Shared Postgres helpers for the loaders and the API.

Connects with the Supabase service role / direct Postgres URL, which bypasses RLS.
Never import this from anything that runs in a browser context.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill it in "
            "(Supabase dashboard -> Project Settings -> Database -> Connection string)."
        )
    return url


@contextmanager
def connect(autocommit: bool = False):
    """Yield a psycopg connection, closing it on exit."""
    conn = psycopg.connect(database_url(), autocommit=autocommit)
    try:
        yield conn
    finally:
        conn.close()


def to_pgvector(values) -> str:
    """Render a float sequence as a pgvector/halfvec literal.

    Passed as text and cast in SQL (`%s::halfvec`), which avoids depending on the
    pgvector client extension just to write a few columns.
    """
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"


def drop_vector_indexes(conn) -> None:
    """Drop HNSW indexes before a bulk load.

    Building the graph incrementally across 44k inserts is dramatically slower than
    loading first and indexing once at the end.
    """
    with conn.cursor() as cur:
        cur.execute("drop index if exists public.jobs_embedding_hnsw")
        cur.execute("drop index if exists public.seeker_embedding_hnsw")
    conn.commit()


def create_vector_indexes(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "create index if not exists jobs_embedding_hnsw on public.jobs "
            "using hnsw (embedding halfvec_cosine_ops) with (m = 16, ef_construction = 64)"
        )
        cur.execute(
            "create index if not exists seeker_embedding_hnsw on public.seeker_profiles "
            "using hnsw (embedding halfvec_cosine_ops) with (m = 16, ef_construction = 64)"
        )
    conn.commit()


def table_sizes(conn) -> list[tuple[str, str]]:
    """Report real on-disk size per relation.

    The free-tier 500 MB cap counts indexes, so this is the number that matters --
    estimates of HNSW overhead are unreliable enough that we measure instead.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            select relname,
                   pg_size_pretty(pg_total_relation_size(c.oid)) as total
            from pg_class c
            join pg_namespace n on n.oid = c.relnamespace
            where n.nspname = 'public' and c.relkind in ('r', 'i')
            order by pg_total_relation_size(c.oid) desc
            """
        )
        return cur.fetchall()
