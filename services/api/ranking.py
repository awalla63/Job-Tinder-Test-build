"""Deck construction: hard filters -> vector ANN -> LightGBM rerank.

Both decks go through here. The seeker deck ranks jobs for one resume; the employer deck
ranks candidates for one posting. Keeping them in one module means the two sides cannot
drift apart in how they filter, score or paginate.

Null-tolerance is the rule that matters most in this file. Only 29% of the posting corpus
carries a salary, 76% an experience level. Every filter below is written so that a NULL
on the *job* passes, and a NULL preference on the *survey* disables that filter entirely.
Writing `normalized_salary >= %s` instead of the null-tolerant form silently discards 71%
of the corpus and the deck just looks thin -- nothing errors.
"""
from __future__ import annotations

import numpy as np

from services.api.config import get_settings


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(v):.6f}" for v in vector) + "]"


def fetch_job_candidates(
    conn,
    resume_embedding: np.ndarray,
    survey: dict | None,
    actor_user_id: str,
    limit: int,
) -> list[dict]:
    """Hard-filter the corpus, then pull the nearest `limit` postings by cosine."""
    survey = survey or {}
    clauses = ["j.embedding is not null"]
    params: list = []

    min_salary = survey.get("min_salary")
    if min_salary:
        # NULL salary passes: an unknown salary is not evidence of a low one.
        clauses.append("(j.normalized_salary is null or j.normalized_salary >= %s)")
        params.append(min_salary)

    remote_pref = survey.get("remote_pref")
    if remote_pref == "remote":
        clauses.append("j.remote_allowed")
    elif remote_pref == "onsite":
        clauses.append("not j.remote_allowed")

    for column, key in (
        ("formatted_work_type", "work_types"),
        ("formatted_experience_level", "experience_levels"),
        ("state", "states"),
    ):
        values = survey.get(key)
        if values:
            # NULL on the job passes -- a posting that omits its work type should not be
            # excluded from every filtered deck forever.
            clauses.append(f"(j.{column} is null or j.{column} = any(%s))")
            params.append(list(values))

    # Never show something this user already swiped on.
    clauses.append(
        "not exists (select 1 from public.swipes s "
        "where s.actor_user_id = %s and s.target_type = 'job' and s.target_id = j.id)"
    )
    params.append(actor_user_id)

    sql = f"""
        select j.id, j.title, j.company_name, j.description, j.location, j.state,
               j.normalized_salary, j.pay_period, j.formatted_work_type,
               j.formatted_experience_level, j.remote_allowed, j.application_url,
               j.source::text, j.employer_id, j.embedding::text
        from public.jobs j
        where {' and '.join(clauses)}
        order by j.embedding <=> %s::halfvec
        limit %s
    """
    params.extend([_vector_literal(resume_embedding), limit])

    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [c.name for c in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def fetch_candidate_profiles(
    conn,
    job_embedding: np.ndarray,
    actor_user_id: str,
    limit: int,
) -> list[dict]:
    """Nearest seeker profiles to a posting, for the employer deck."""
    sql = """
        select p.id, p.user_id, p.is_demo, p.display_name, p.headline, p.resume_text,
               p.years_experience, p.predicted_category, p.embedding::text
        from public.seeker_profiles p
        where p.embedding is not null
          and not exists (
              select 1 from public.swipes s
              where s.actor_user_id = %s and s.target_type = 'seeker' and s.target_id = p.id
          )
        order by p.embedding <=> %s::halfvec
        limit %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (actor_user_id, _vector_literal(job_embedding), limit))
        columns = [c.name for c in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def parse_vector(text: str | None) -> np.ndarray | None:
    """halfvec comes back as '[0.1,0.2,...]'."""
    if not text:
        return None
    return np.fromstring(text.strip()[1:-1], sep=",", dtype=np.float32)


def rank_jobs_for_resume(
    conn,
    scorer,
    resume_text: str,
    resume_embedding: np.ndarray,
    survey: dict | None,
    actor_user_id: str,
    limit: int,
    resume_years: float | None = None,
) -> list[dict]:
    settings = get_settings()
    candidates = fetch_job_candidates(
        conn, resume_embedding, survey, actor_user_id, settings.ann_candidates
    )
    if not candidates:
        return []

    pairs = []
    for row in candidates:
        vector = parse_vector(row.pop("embedding"))
        text = " ".join(p for p in (row["title"], row["company_name"], row["description"]) if p)
        pairs.append((text, vector if vector is not None else resume_embedding * 0))

    scores = scorer.score_one_resume_many_jobs(
        resume_text, resume_embedding, pairs, resume_years
    )

    from services.api.scoring import explain

    for row, (text, _), score in zip(candidates, pairs, scores):
        row["score"] = round(float(score), 1)
        row["reasons"] = explain(resume_text, text, float(score))

    candidates.sort(key=lambda r: -r["score"])
    return candidates[:limit]


def rank_candidates_for_job(
    conn,
    scorer,
    job_text: str,
    job_embedding: np.ndarray,
    actor_user_id: str,
    limit: int,
) -> list[dict]:
    settings = get_settings()
    rows = fetch_candidate_profiles(
        conn, job_embedding, actor_user_id, settings.ann_candidates
    )
    if not rows:
        return []

    from services.api.scoring import explain

    scored = []
    resumes: list[tuple[str, np.ndarray, float | None]] = []
    for row in rows:
        vector = parse_vector(row.pop("embedding"))
        if vector is None:
            continue
        years = row.get("years_experience")
        resumes.append((row["resume_text"] or "", vector,
                        float(years) if years is not None else None))
        scored.append(row)

    if not scored:
        return []

    scores = scorer.score_many_resumes_one_job(job_text, job_embedding, resumes)

    for row, score in zip(scored, scores):
        row["score"] = round(float(score), 1)
        row["reasons"] = explain(row["resume_text"], job_text, float(score))
        # Never ship full resume text to the browser; the card shows a preview.
        row["resume_preview"] = (row.pop("resume_text") or "")[:400]

    scored.sort(key=lambda r: -r["score"])
    return scored[:limit]
