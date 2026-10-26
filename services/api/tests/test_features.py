"""Feature extraction and the fit-score collapse."""
from __future__ import annotations

import numpy as np

from ml.features import (
    ALL_FEATURE_NAMES,
    FEATURE_SETS,
    PAIRWISE_FEATURES,
    SINGLE_SIDE_FEATURES,
    JobSide,
    ResumeSide,
    build_matrix,
    build_row,
    fit_score,
    required_years,
    resume_years,
    select_columns,
    seniority,
)


def test_required_years_takes_the_lowest_stated_bar():
    # "3-5 years" and "5+ preferred, 3 required" both mean the bar is 3.
    assert required_years("We need 3-5 years of experience") == 3.0
    assert required_years("5+ years preferred, 3 years required") == 3.0
    assert np.isnan(required_years("No experience mentioned"))


def test_resume_years_spans_earliest_to_latest():
    assert resume_years("Worked 2015 to 2021 at Acme") == 6.0
    # A single year is not a span.
    assert np.isnan(resume_years("Graduated 2019"))
    # Implausible spans are rejected rather than passed through.
    assert np.isnan(resume_years("1890 and 2024"))


def test_seniority_takes_the_highest_signal():
    assert seniority("Senior Director of Engineering") == 5.0
    assert seniority("Senior Analyst") == 3.0
    assert np.isnan(seniority("Baker"))


def test_explicit_seniority_modifiers_outrank_role_nouns():
    """"Junior Analyst" is a junior role.

    Scoring it off "analyst" (mid-level) would push genuinely entry-level candidates
    down the deck for exactly the postings they should be seeing.
    """
    assert seniority("Junior analyst") == 1.0
    assert seniority("Intern, data engineering") == 0.0
    # With no modifier present, the role noun is used as weaker evidence.
    assert seniority("Analyst") == 2.0
    assert seniority("Manager") == 4.0


def test_missing_values_stay_nan_rather_than_becoming_zero():
    """A missing salary or year count must not read as a real zero.

    LightGBM learns a split direction for NaN; a zero asserts something false.
    """
    row = build_row(
        ResumeSide(text="Nothing useful here", embedding=None),
        JobSide(text="Also nothing", embedding=None),
    )
    values = dict(zip(ALL_FEATURE_NAMES, row))
    assert np.isnan(values["emb_cosine"])       # no embeddings supplied
    assert np.isnan(values["job_required_years"])
    assert np.isnan(values["years_gap"])


def test_embedding_cosine_is_a_dot_product_of_unit_vectors():
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1.0
    row = build_row(
        ResumeSide(text="python engineer", embedding=vector),
        JobSide(text="python engineer", embedding=vector),
    )
    assert abs(dict(zip(ALL_FEATURE_NAMES, row))["emb_cosine"] - 1.0) < 1e-6


def test_skill_overlap_counts_shared_lexicon_entries():
    row = build_row(
        ResumeSide(text="Expert in Python, SQL and Kubernetes", embedding=None),
        JobSide(text="Looking for Python and SQL skills", embedding=None),
    )
    values = dict(zip(ALL_FEATURE_NAMES, row))
    assert values["skill_overlap_count"] >= 2
    assert 0 < values["skill_jaccard"] <= 1


def test_fit_score_is_monotonic_in_the_ordinal_label():
    scores = fit_score(np.array([
        [1.0, 0.0, 0.0],   # certain No Fit
        [0.0, 1.0, 0.0],   # certain Potential Fit
        [0.0, 0.0, 1.0],   # certain Good Fit
    ]))
    assert scores[0] == 0.0
    assert scores[1] == 50.0
    assert scores[2] == 100.0
    assert scores[0] < scores[1] < scores[2]


def test_build_matrix_rejects_misaligned_inputs():
    import pytest

    with pytest.raises(ValueError):
        build_matrix([ResumeSide(text="a")], [JobSide(text="b"), JobSide(text="c")])


def test_length_proxies_are_excluded_from_the_shipped_feature_set():
    """The default feature set must not contain document-length features.

    With only 643 resumes and 351 JDs in the training data, a per-document constant
    like length acts as a document ID: the model memorises that document's base rate
    instead of learning fit, and the memorisation cannot transfer to unseen postings.
    The ablation in ml/train.py measured this -- dropping them raised cold-JD AUC from
    0.6775 to 0.6934.
    """
    shipped = FEATURE_SETS["pairwise_plus"]
    assert "resume_len_log" not in shipped
    assert "job_len_log" not in shipped
    assert set(PAIRWISE_FEATURES).issubset(shipped)


def test_select_columns_preserves_requested_order():
    matrix = np.arange(len(ALL_FEATURE_NAMES), dtype=np.float32).reshape(1, -1)
    picked = select_columns(matrix, ["tfidf_cosine", "emb_cosine"])
    assert picked[0, 0] == ALL_FEATURE_NAMES.index("tfidf_cosine")
    assert picked[0, 1] == ALL_FEATURE_NAMES.index("emb_cosine")


def test_feature_families_partition_cleanly():
    assert not set(PAIRWISE_FEATURES) & set(SINGLE_SIDE_FEATURES)
    assert set(PAIRWISE_FEATURES) | set(SINGLE_SIDE_FEATURES) == set(ALL_FEATURE_NAMES)
