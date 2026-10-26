"""Scoring fallback, resume parsing, and the loader/connector mappings."""
from __future__ import annotations

import numpy as np
import pytest

from data.load_postings import as_bool, as_float, parse_state
from data.usajobs_client import map_record
from ml.features import JobSide, ResumeSide
from services.api.resume_parse import UnsupportedResume, extract_text
from services.api.scoring import Scorer, explain


# --- scoring fallback ------------------------------------------------------

def test_scorer_falls_back_to_cosine_without_a_model():
    """No model file must degrade ranking, not break the app."""
    scorer = Scorer()
    assert scorer.ready is False

    a = np.zeros(384, dtype=np.float32); a[0] = 1.0
    b = np.zeros(384, dtype=np.float32); b[0] = 1.0
    c = np.zeros(384, dtype=np.float32); c[1] = 1.0

    scores = scorer.score(
        [ResumeSide(text="x", embedding=a), ResumeSide(text="x", embedding=a)],
        [JobSide(text="y", embedding=b), JobSide(text="y", embedding=c)],
    )
    assert scores[0] == pytest.approx(100.0)   # identical direction
    assert scores[1] == pytest.approx(50.0)    # orthogonal
    assert all(0 <= s <= 100 for s in scores)


def test_score_rejects_misaligned_batches():
    with pytest.raises(ValueError):
        Scorer().score([ResumeSide(text="a")], [])


def test_explain_mentions_shared_skills_and_experience():
    reasons = explain(
        "Senior engineer with Python and Kubernetes, 2014 to 2024",
        "We need 5 years of Python and Kubernetes experience",
        82.0,
    )
    assert any("Python" in r or "python" in r.lower() for r in reasons)
    assert any("yrs" in r for r in reasons)
    assert len(reasons) <= 3


# --- resume parsing --------------------------------------------------------

def test_txt_resume_is_read():
    body = ("Experienced data analyst with SQL and Python. " * 10).encode()
    assert "data analyst" in extract_text(body, "resume.txt")


def test_unknown_extension_is_rejected():
    with pytest.raises(UnsupportedResume):
        extract_text(b"x" * 500, "resume.rtf")


def test_legacy_doc_gets_an_actionable_message():
    with pytest.raises(UnsupportedResume, match="docx"):
        extract_text(b"x" * 500, "resume.doc")


def test_unreadable_file_is_rejected_rather_than_stored_empty():
    """An empty resume embeds to noise and yields an empty deck.

    Failing the upload is far clearer to the user than a silently broken account.
    """
    with pytest.raises(UnsupportedResume):
        extract_text(b"tiny", "resume.txt")


# --- postings loader -------------------------------------------------------

def test_state_parsing_handles_abbreviations_and_full_names():
    assert parse_state("Charlotte, NC") == "NC"
    assert parse_state("North Carolina, United States") == "NC"
    assert parse_state("") is None


def test_views_parse_as_float_not_int():
    """The CSV stores counts as "2.0"; int() on that raises."""
    assert as_float("2.0") == 2.0
    assert as_float("") is None
    assert as_float("n/a") is None


def test_remote_flag_parsing():
    assert as_bool("1.0") is True
    assert as_bool("true") is True
    assert as_bool("") is False


# --- USAJOBS mapping -------------------------------------------------------

def _descriptor(**overrides):
    base = {
        "PositionTitle": "Data Scientist",
        "PositionURI": "https://www.usajobs.gov/job/123",
        "PositionLocationDisplay": "Charlotte, North Carolina",
        "PositionLocation": [{"CountrySubDivisionCode": "NC"}],
        "PositionRemuneration": [
            {"MinimumRange": "100000", "MaximumRange": "120000", "RateIntervalCode": "PA"}
        ],
        "PublicationStartDate": "2026-10-01",
        "OrganizationName": "Department of Commerce",
        "UserArea": {"Details": {"JobSummary": "Analyse data for the agency.",
                                 "TeleworkEligible": True}},
    }
    base.update(overrides)
    return {"MatchedObjectId": "123456", "MatchedObjectDescriptor": base}


def test_usajobs_record_maps_onto_job_columns():
    record = map_record(_descriptor())
    assert record["external_id"] == "123456"
    assert record["title"] == "Data Scientist"
    assert record["description"].startswith("Analyse data")
    # ToS requires sending applicants to the official announcement.
    assert record["application_url"] == "https://www.usajobs.gov/job/123"
    assert record["state"] == "NC"
    assert record["remote_allowed"] is True
    assert record["normalized_salary"] == pytest.approx(110000.0)


def test_hourly_pay_is_annualised():
    record = map_record(_descriptor(PositionRemuneration=[
        {"MinimumRange": "50", "MaximumRange": "50", "RateIntervalCode": "PH"}
    ]))
    assert record["pay_period"] == "HOURLY"
    assert record["normalized_salary"] == pytest.approx(50 * 2080)


def test_record_without_a_summary_is_dropped():
    assert map_record(_descriptor(UserArea={"Details": {}})) is None


def test_missing_salary_maps_to_none_not_zero():
    record = map_record(_descriptor(PositionRemuneration=[]))
    assert record["normalized_salary"] is None
