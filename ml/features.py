"""Pairwise (resume, job) features for the fit model.

This module is imported verbatim by `ml/train.py` and by the API's ranking path. It is
never reimplemented on the serving side: a feature that is computed one way in training
and another way in serving produces ranking bugs that are nearly invisible in tests and
obvious only as bad recommendations.

---------------------------------------------------------------------------
Why survey preferences are NOT features here
---------------------------------------------------------------------------
The labelled training data (cnamuangtoun/resume-job-description-fit) contains exactly
three columns: resume_text, job_description_text, label. There is no salary, no work
type, no location, no remote flag. So nothing derived from those fields can be learned
from it -- a "salary gap" feature would be present at serving time and absent at
training time, which is the textbook recipe for train/serve skew.

Preferences are therefore handled where they belong: as **hard filters in SQL** before
retrieval (see services/api/ranking.py). The learned model answers one question only --
"does this person's background fit this job's requirements?" -- using signals that exist
identically in training and in production.

---------------------------------------------------------------------------
Known approximation
---------------------------------------------------------------------------
Stored job embeddings cover `title + company + description`, while the training pairs
only have a description. The title is a small fraction of a ~4,000-character document and
is further diluted by chunk-mean-pooling, so the shift is minor -- but it is real, and
the cold-JD evaluation in train.py is what would surface it if it ever stopped being minor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ml import skills

ALL_FEATURE_NAMES: list[str] = [
    "emb_cosine",
    "tfidf_cosine",
    "skill_overlap_count",
    "skill_jaccard",
    "skill_coverage",          # share of the job's skills the resume covers
    "skill_idf_weighted",
    "resume_years",
    "job_required_years",
    "years_gap",               # resume_years - job_required_years; NaN if either unknown
    "resume_seniority",
    "job_seniority",
    "seniority_gap",
    "resume_len_log",
    "job_len_log",
    "token_overlap_jaccard",
]

# Features computed from BOTH sides of the pair. These are the only ones that can
# describe a *relationship*; everything else describes one document in isolation.
PAIRWISE_FEATURES: list[str] = [
    "emb_cosine",
    "tfidf_cosine",
    "skill_overlap_count",
    "skill_jaccard",
    "skill_coverage",
    "skill_idf_weighted",
    "years_gap",
    "seniority_gap",
    "token_overlap_jaccard",
]

# Single-side features are a liability on this dataset, not an asset. With only 643
# resumes and 351 job descriptions -- median 16 rows per JD -- a value that is constant
# across every row of a document (its length, its stated seniority) works as a document
# identifier, and the model learns that document's base rate instead of learning fit.
# That memorisation cannot transfer to the 44k unseen postings we actually serve.
# `resume_len_log` and `job_len_log` are the worst offenders: document length has no
# causal bearing on whether a person fits a role, yet it ranked as the top feature by
# gain when included. See models/metrics.json for the ablation that settled this.
SINGLE_SIDE_FEATURES: list[str] = [
    "resume_years", "job_required_years", "resume_seniority", "job_seniority",
    "resume_len_log", "job_len_log",
]

FEATURE_SETS: dict[str, list[str]] = {
    "all": ALL_FEATURE_NAMES,
    "pairwise": PAIRWISE_FEATURES,
    # Pairwise plus the single-side signals that are at least semantically defensible
    # (a role demanding 15 years really is harder to fit), but no length proxies.
    "pairwise_plus": PAIRWISE_FEATURES + ["job_required_years", "job_seniority"],
}

# Default kept as the full list so build_row/build_matrix stay backwards-compatible;
# train.py and the API select a subset by name via `select_columns`.
FEATURE_NAMES = ALL_FEATURE_NAMES


def select_columns(matrix: np.ndarray, names: list[str]) -> np.ndarray:
    """Slice a full feature matrix down to `names`, preserving their order."""
    index = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}
    return matrix[:, [index[n] for n in names]]

LABELS = ["No Fit", "Potential Fit", "Good Fit"]
LABEL_TO_INT = {name: i for i, name in enumerate(LABELS)}

# "5+ years", "minimum of 3 years", "3-5 years of experience"
_REQ_YEARS_RE = re.compile(
    r"(\d{1,2})\s*(?:\+|plus)?\s*(?:-|to|–)?\s*(?:\d{1,2})?\s*\+?\s*year", re.I
)
_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
_WORD_RE = re.compile(r"[a-z][a-z0-9+#.\-]{2,}")

# Explicit seniority words. These are deliberate statements about level and they win
# outright: "Junior Analyst" is a junior role, even though "analyst" on its own reads as
# mid-level, and scoring it as mid-level would push entry-level candidates down the deck.
_SENIORITY_MODIFIERS = {
    "intern": 0, "internship": 0, "trainee": 0,
    "junior": 1, "entry level": 1, "entry-level": 1, "jr.": 1,
    "mid-level": 2,
    "senior": 3, "sr.": 3, "lead": 3,
    "principal": 4, "staff": 4, "head of": 4,
    "director": 5, "vp": 5, "vice president": 5, "executive": 5,
    "chief": 6,
}

# Role nouns, used only when no explicit modifier is present. Weaker evidence: a title
# can carry one of these at almost any level.
_SENIORITY_ROLES = {
    "assistant": 1, "associate": 1,
    "specialist": 2, "analyst": 2, "engineer": 2, "coordinator": 2,
    "supervisor": 3, "manager": 4,
}

_STOP = {
    "the", "and", "for", "with", "you", "are", "our", "will", "that", "this", "have",
    "from", "your", "all", "can", "has", "not", "but", "they", "their", "work", "job",
    "role", "who", "any", "may", "was", "its", "more", "other", "than", "such", "them",
    "what", "when", "which", "also", "been", "were", "would", "about", "into", "over",
}


# ---------------------------------------------------------------------------
# Side containers
# ---------------------------------------------------------------------------
@dataclass
class ResumeSide:
    text: str
    embedding: np.ndarray | None = None
    years_experience: float | None = None


@dataclass
class JobSide:
    text: str
    embedding: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Scalar extractors
# ---------------------------------------------------------------------------
def required_years(text: str) -> float:
    """Smallest plausible "N years" requirement stated in a job description.

    Takes the minimum rather than the max: "3-5 years" and "5+ years preferred, 3
    required" both mean the bar is 3.
    """
    if not text:
        return np.nan
    values = [int(m) for m in _REQ_YEARS_RE.findall(text) if 0 < int(m) <= 40]
    return float(min(values)) if values else np.nan


def resume_years(text: str) -> float:
    """Span between earliest and latest year mentioned on the resume."""
    if not text:
        return np.nan
    years = [int(y) for y in _YEAR_RE.findall(text)]
    if len(years) < 2:
        return np.nan
    span = max(years) - min(years)
    return float(span) if 0 < span <= 45 else np.nan


def seniority(text: str) -> float:
    """Seniority signal on a 0 (intern) - 6 (chief) scale, or NaN if none is present.

    Explicit modifiers take precedence over role nouns; within each tier the highest
    signal wins, so "Senior Director" reads as director rather than senior.
    """
    if not text:
        return np.nan
    low = text.lower()

    modifiers = [level for term, level in _SENIORITY_MODIFIERS.items() if term in low]
    if modifiers:
        return float(max(modifiers))

    roles = [level for term, level in _SENIORITY_ROLES.items() if term in low]
    return float(max(roles)) if roles else np.nan


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall((text or "").lower()) if t not in _STOP}


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------
def build_row(
    resume: ResumeSide,
    job: JobSide,
    *,
    tfidf_cosine: float = np.nan,
    skill_idf: dict[str, float] | None = None,
) -> list[float]:
    """Produce one feature vector, ordered to match FEATURE_NAMES.

    Missing values are returned as NaN rather than imputed. LightGBM handles NaN
    natively and learns a split direction for it; imputing a zero would instead assert
    something false (a resume with "0 years of experience", a job requiring none).
    """
    r_skills = skills.extract(resume.text)
    j_skills = skills.extract(job.text)
    shared = r_skills & j_skills
    union = r_skills | j_skills

    if skill_idf:
        idf_weighted = sum(skill_idf.get(s, 1.0) for s in shared)
    else:
        idf_weighted = float(len(shared))

    emb_cos = np.nan
    if resume.embedding is not None and job.embedding is not None:
        emb_cos = float(np.dot(resume.embedding, job.embedding))

    r_years = resume.years_experience if resume.years_experience is not None else resume_years(resume.text)
    j_years = required_years(job.text)
    r_sen = seniority(resume.text)
    j_sen = seniority(job.text)

    r_tokens, j_tokens = _tokens(resume.text), _tokens(job.text)
    token_union = r_tokens | j_tokens

    return [
        emb_cos,
        float(tfidf_cosine),
        float(len(shared)),
        len(shared) / len(union) if union else np.nan,
        len(shared) / len(j_skills) if j_skills else np.nan,
        float(idf_weighted),
        float(r_years) if r_years is not None else np.nan,
        j_years,
        (float(r_years) - j_years) if (r_years is not None and not np.isnan(r_years)
                                       and not np.isnan(j_years)) else np.nan,
        r_sen,
        j_sen,
        (r_sen - j_sen) if not (np.isnan(r_sen) or np.isnan(j_sen)) else np.nan,
        float(np.log1p(len(resume.text or ""))),
        float(np.log1p(len(job.text or ""))),
        len(r_tokens & j_tokens) / len(token_union) if token_union else np.nan,
    ]


def build_matrix(
    resumes: list[ResumeSide],
    jobs: list[JobSide],
    *,
    tfidf_cosines: np.ndarray | None = None,
    skill_idf: dict[str, float] | None = None,
) -> np.ndarray:
    """Feature matrix for aligned resume/job lists (element i pairs with element i)."""
    if len(resumes) != len(jobs):
        raise ValueError(f"length mismatch: {len(resumes)} resumes vs {len(jobs)} jobs")
    cosines = tfidf_cosines if tfidf_cosines is not None else np.full(len(resumes), np.nan)
    rows = [
        build_row(r, j, tfidf_cosine=c, skill_idf=skill_idf)
        for r, j, c in zip(resumes, jobs, cosines)
    ]
    return np.asarray(rows, dtype=np.float32)


def compute_skill_idf(documents: list[str]) -> dict[str, float]:
    """IDF over the skill lexicon.

    Downweights skills that appear everywhere ("communication") relative to ones that
    actually discriminate ("kubernetes", "icd-10").
    """
    n = max(1, len(documents))
    counts: dict[str, int] = {}
    for doc in documents:
        for skill in skills.extract(doc):
            counts[skill] = counts.get(skill, 0) + 1
    return {s: float(np.log(n / (1 + c)) + 1.0) for s, c in counts.items()}


def fit_score(probabilities: np.ndarray) -> np.ndarray:
    """Collapse 3-class probabilities into one 0-100 score.

    Ordinal expectation: 0.5 * P(Potential Fit) + 1.0 * P(Good Fit). Using the expected
    label keeps the ordering the annotation encodes, instead of discarding the middle
    class by taking P(Good Fit) alone.
    """
    probabilities = np.atleast_2d(probabilities)
    weights = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    return np.clip(probabilities @ weights, 0.0, 1.0) * 100.0
