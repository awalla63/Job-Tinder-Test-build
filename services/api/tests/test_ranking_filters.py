"""Hard-filter behaviour.

The null-tolerance tests here are the important ones. Only 29% of the posting corpus
carries a salary and 76% an experience level, so a filter written the obvious way
(`normalized_salary >= %s`) silently drops 71% of the corpus without raising anything --
the deck just looks thin. These tests pin the null-tolerant form in place.
"""
from __future__ import annotations

import numpy as np
import pytest

from services.api.ranking import fetch_job_candidates, parse_vector


class FakeCursor:
    """Captures the SQL and parameters instead of executing them."""

    def __init__(self):
        self.sql = ""
        self.params = None
        self.description = []

    def execute(self, sql, params=None):
        self.sql = sql
        self.params = params

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self):
        self.cursor_obj = FakeCursor()

    def cursor(self):
        return self.cursor_obj


@pytest.fixture
def embedding():
    return np.ones(384, dtype=np.float32) / 20.0


def run(survey, embedding):
    conn = FakeConn()
    fetch_job_candidates(conn, embedding, survey, "user-1", 200)
    return conn.cursor_obj


def where_clause(cursor) -> str:
    """Just the filter predicates.

    Every filtered column also appears in the SELECT list, so asserting against the
    whole statement would pass trivially for 'is this column filtered on?'.
    """
    sql = " ".join(cursor.sql.split())
    return sql[sql.index(" where ") : sql.index(" order by ")]


def test_salary_filter_lets_null_salary_through(embedding):
    cursor = run({"min_salary": 90000}, embedding)
    assert "normalized_salary is null or j.normalized_salary >= %s" in cursor.sql
    assert 90000 in cursor.params


def test_no_salary_clause_when_unset(embedding):
    assert "normalized_salary" not in where_clause(run({}, embedding))


def test_work_type_and_level_filters_let_nulls_through(embedding):
    cursor = run({"work_types": ["Full-time"], "experience_levels": ["Entry level"]}, embedding)
    assert "j.formatted_work_type is null or j.formatted_work_type = any(%s)" in cursor.sql
    assert "j.formatted_experience_level is null or j.formatted_experience_level = any(%s)" in cursor.sql


def test_remote_preference_is_strict_both_ways(embedding):
    assert "j.remote_allowed" in where_clause(run({"remote_pref": "remote"}, embedding))
    assert "not j.remote_allowed" in where_clause(run({"remote_pref": "onsite"}, embedding))
    # "any" is normalised to no filter at all, not to a clause matching everything.
    assert "remote_allowed" not in where_clause(run({"remote_pref": "any"}, embedding))


def test_already_swiped_jobs_are_excluded(embedding):
    cursor = run({}, embedding)
    assert "not exists" in cursor.sql
    assert "s.target_type = 'job'" in cursor.sql
    assert "user-1" in cursor.params


def test_orders_by_cosine_distance_operator(embedding):
    cursor = run({}, embedding)
    # <=> is cosine; <-> would be L2 and <#> negative inner product, either of which
    # would rank wrongly against a cosine-trained encoder.
    assert "j.embedding <=> %s::halfvec" in cursor.sql


def test_requires_an_embedding(embedding):
    assert "j.embedding is not null" in run({}, embedding).sql


def test_parse_vector_roundtrip():
    assert parse_vector("[0.5,-0.25,1.0]").tolist() == [0.5, -0.25, 1.0]
    assert parse_vector(None) is None
    assert parse_vector("") is None
