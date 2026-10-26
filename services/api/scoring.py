"""Model loading and pair scoring.

Loaded once at process start (see main.py's lifespan) and reused for every request.

If no trained model is present, this degrades to embedding cosine rather than failing.
That is a deliberate product decision, not a stopgap: the ranking gets worse, but the
app stays completely usable, and `ml/train.py` tells you to delete the model file if it
cannot beat the cosine baseline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import normalize

from ml.features import ALL_FEATURE_NAMES, JobSide, ResumeSide, build_matrix, fit_score
from services.api.config import get_settings

log = logging.getLogger(__name__)


@dataclass
class ScoredPair:
    index: int
    score: float
    reasons: list[str]


class Scorer:
    def __init__(self) -> None:
        self.booster = None
        self.vectorizer = None
        self.skill_idf: dict[str, float] = {}
        self.feature_names: list[str] = ALL_FEATURE_NAMES
        self._column_index: list[int] = list(range(len(ALL_FEATURE_NAMES)))

    # -- lifecycle --------------------------------------------------------------

    def load(self) -> None:
        settings = get_settings()
        if not settings.fit_model_path.exists():
            log.warning(
                "No model at %s - serving embedding cosine only. Run ml/train.py.",
                settings.fit_model_path,
            )
            return

        import lightgbm as lgb

        self.booster = lgb.Booster(model_file=str(settings.fit_model_path))
        # Pin to a single thread and scale with uvicorn workers instead. Letting each
        # request fan out across cores oversubscribes badly under concurrency and
        # measurably worsens tail latency.
        self.booster.params["num_threads"] = 1

        if settings.fit_artifacts_path.exists():
            import joblib

            artifacts = joblib.load(settings.fit_artifacts_path)
            self.vectorizer = artifacts.get("vectorizer")
            self.skill_idf = artifacts.get("skill_idf", {})
            self.feature_names = artifacts.get("feature_names", ALL_FEATURE_NAMES)

        # The saved model was trained on a specific column subset in a specific order.
        # Resolve that to indices once here so every request slices identically -- a
        # silent mismatch would mis-rank everything while looking perfectly healthy.
        lookup = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}
        missing = [n for n in self.feature_names if n not in lookup]
        if missing:
            raise RuntimeError(
                f"Model expects unknown features {missing}. ml/features.py and the "
                f"saved artifacts are out of sync; retrain with ml/train.py."
            )
        self._column_index = [lookup[n] for n in self.feature_names]
        log.info("Loaded fit model with %d features", len(self.feature_names))

    @property
    def ready(self) -> bool:
        return self.booster is not None

    # -- scoring ----------------------------------------------------------------

    def _tfidf_cosines(self, resume_texts: list[str], job_texts: list[str]) -> np.ndarray:
        if self.vectorizer is None:
            return np.full(len(resume_texts), np.nan)
        R = normalize(self.vectorizer.transform(resume_texts))
        J = normalize(self.vectorizer.transform(job_texts))
        return np.asarray(R.multiply(J).sum(axis=1)).ravel()

    def score(
        self,
        resumes: list[ResumeSide],
        jobs: list[JobSide],
    ) -> np.ndarray:
        """Score aligned (resume, job) pairs in one batch. Returns 0-100 per pair.

        Both decks funnel through here, so the whole batch gets a single TF-IDF
        transform and a single Booster.predict -- scoring 200 pairs is 1-5 ms, where
        looping one pair at a time would pay the vectoriser cost 200 times over.
        """
        if not resumes:
            return np.zeros(0, dtype=np.float32)
        if len(resumes) != len(jobs):
            raise ValueError(f"length mismatch: {len(resumes)} vs {len(jobs)}")

        if not self.ready:
            # Cosine fallback, mapped from [-1, 1] onto [0, 100].
            cosines = np.array([
                float(np.dot(r.embedding, j.embedding))
                if r.embedding is not None and j.embedding is not None else 0.0
                for r, j in zip(resumes, jobs)
            ])
            return np.clip((cosines + 1.0) / 2.0, 0.0, 1.0) * 100.0

        matrix = build_matrix(
            resumes,
            jobs,
            tfidf_cosines=self._tfidf_cosines([r.text for r in resumes],
                                              [j.text for j in jobs]),
            skill_idf=self.skill_idf,
        )
        probabilities = self.booster.predict(matrix[:, self._column_index])
        return fit_score(probabilities)

    def score_one_resume_many_jobs(
        self,
        resume_text: str,
        resume_embedding: np.ndarray,
        jobs: list[tuple[str, np.ndarray]],
        resume_years: float | None = None,
    ) -> np.ndarray:
        """Seeker deck: one resume against many postings."""
        side = ResumeSide(text=resume_text, embedding=resume_embedding,
                          years_experience=resume_years)
        return self.score([side] * len(jobs),
                          [JobSide(text=t, embedding=v) for t, v in jobs])

    def score_many_resumes_one_job(
        self,
        job_text: str,
        job_embedding: np.ndarray,
        resumes: list[tuple[str, np.ndarray, float | None]],
    ) -> np.ndarray:
        """Employer deck: many candidates against one posting.

        Scored in the same direction the model was trained in (resume as the query
        side, job as the passage side) so the two decks produce comparable numbers.
        """
        job_side = JobSide(text=job_text, embedding=job_embedding)
        return self.score(
            [ResumeSide(text=t, embedding=v, years_experience=y) for t, v, y in resumes],
            [job_side] * len(resumes),
        )


def explain(resume_text: str, job_text: str, score: float) -> list[str]:
    """Short, honest reasons shown on the card.

    Derived from the same signals the model consumes, so what the user reads lines up
    with what actually drove the ranking.
    """
    from ml import skills
    from ml.features import required_years, resume_years as _resume_years

    reasons: list[str] = []

    shared = skills.extract(resume_text) & skills.extract(job_text)
    if shared:
        top = sorted(shared, key=len, reverse=True)[:3]
        reasons.append("Shared skills: " + ", ".join(top))

    needed = required_years(job_text)
    have = _resume_years(resume_text)
    if not np.isnan(needed) and have is not None and not np.isnan(have):
        if have >= needed:
            reasons.append(f"Experience: ~{have:.0f} yrs vs ~{needed:.0f} asked")
        else:
            reasons.append(f"Asks for ~{needed:.0f} yrs; resume suggests ~{have:.0f}")

    if score >= 70:
        reasons.append("Strong overall match to the description")
    elif score >= 45:
        reasons.append("Partial match to the description")
    else:
        reasons.append("Limited overlap with the description")

    return reasons[:3]
