"""The mutual-match rule.

Two paths, and getting them confused is the difference between a working product and a
dead one:

  * Sourced jobs (LinkedIn seed, USAJOBS) have no employer account. Requiring a mutual
    like would mean no seeker ever matches anything, so a like alone creates the match.
  * Employer-posted jobs are the genuine two-sided case and need both sides.
"""
from __future__ import annotations

import pytest

from services.api.routers.matching import _employer_liked_seeker, _seeker_liked_job


class ScriptedCursor:
    """Returns queued results in order and records every statement executed."""

    def __init__(self, results):
        self._results = list(results)
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self._results.pop(0) if self._results else None


def test_sourced_job_matches_on_a_single_like():
    cursor = ScriptedCursor([
        ("https://apply.example/123", None, None),  # employer_id is NULL => sourced
        ("match-1",),
    ])
    result = _seeker_liked_job(cursor, "seeker-1", "job-1")

    assert result.matched is True
    assert result.application_url == "https://apply.example/123"
    assert result.match_id == "match-1"
    # It must not have gone looking for a reciprocal employer like.
    assert not any("target_type = 'seeker'" in sql for sql, _ in cursor.executed)


def test_employer_job_does_not_match_without_a_reciprocal_like():
    cursor = ScriptedCursor([
        ("https://apply.example/9", "employer-1", "employer-user-1"),
        None,  # no reciprocal like from the employer
    ])
    result = _seeker_liked_job(cursor, "seeker-1", "job-1")

    assert result.recorded is True
    assert result.matched is False
    assert result.match_id is None
    assert not any("insert into public.matches" in sql for sql, _ in cursor.executed)


def test_employer_job_matches_once_both_sides_have_liked():
    cursor = ScriptedCursor([
        ("https://apply.example/9", "employer-1", "employer-user-1"),
        (1,),           # employer already liked this seeker
        ("match-2",),
    ])
    result = _seeker_liked_job(cursor, "seeker-1", "job-1")

    assert result.matched is True
    assert result.match_id == "match-2"
    assert any("insert into public.matches" in sql for sql, _ in cursor.executed)


def test_missing_job_is_a_404():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _seeker_liked_job(ScriptedCursor([None]), "seeker-1", "nope")
    assert exc.value.status_code == 404


def test_demo_profiles_can_be_swiped_but_never_match():
    """Resume.csv profiles exist to populate the employer deck on day one.

    They have no auth user behind them, so there is nobody to have liked back -- the
    employer's swipe is recorded, and that is all.
    """
    cursor = ScriptedCursor([(None, True)])  # user_id NULL, is_demo True
    result = _employer_liked_seeker(cursor, "employer-user-1", "demo-profile-1")

    assert result.recorded is True
    assert result.matched is False
    assert not any("insert into public.matches" in sql for sql, _ in cursor.executed)


def test_employer_like_matches_a_real_seeker_who_liked_their_job():
    cursor = ScriptedCursor([
        ("seeker-user-1", False),                      # real profile
        ("job-7", "https://apply.example/7"),          # that seeker liked job-7
        ("match-3",),
    ])
    result = _employer_liked_seeker(cursor, "employer-user-1", "profile-1")

    assert result.matched is True
    assert result.match_id == "match-3"
    assert result.application_url == "https://apply.example/7"


def test_employer_like_without_a_prior_seeker_like_does_not_match():
    cursor = ScriptedCursor([
        ("seeker-user-1", False),
        None,  # the seeker has not liked any of this employer's jobs
    ])
    result = _employer_liked_seeker(cursor, "employer-user-1", "profile-1")

    assert result.matched is False
