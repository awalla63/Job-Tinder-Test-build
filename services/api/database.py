"""Connection pool.

The API connects as the service role, which bypasses RLS. That is correct here: the
backend scores one user's resume against other users' postings and vice versa, which no
single-user policy can express. Authorisation is enforced in the route handlers instead,
off the verified JWT subject -- so every query that touches user data must filter on it
explicitly.
"""
from __future__ import annotations

from contextlib import contextmanager

from psycopg_pool import ConnectionPool

from services.api.config import get_settings

_pool: ConnectionPool | None = None


def init_pool() -> ConnectionPool:
    global _pool
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured; see .env.example")
    _pool = ConnectionPool(settings.database_url, min_size=1, max_size=8, open=True)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def get_connection():
    if _pool is None:
        raise RuntimeError("Connection pool is not initialised")
    with _pool.connection() as conn:
        yield conn
