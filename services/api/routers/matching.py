"""Decks, swipes and matches."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from services.api.auth import CurrentUser, get_current_user
from services.api.database import get_connection
from services.api.ranking import (
    parse_vector,
    rank_candidates_for_job,
    rank_jobs_for_resume,
)
from services.api.schemas import SwipeIn, SwipeOut

log = logging.getLogger(__name__)
router = APIRouter(tags=["matching"])


def _load_survey(cur, user_id: str) -> dict:
    cur.execute(
        "select min_salary, remote_pref, work_types, states, categories, experience_levels "
        "from public.surveys where user_id = %s",
        (user_id,),
    )
    row = cur.fetchone()
    if not row:
        return {}
    keys = ("min_salary", "remote_pref", "work_types", "states", "categories",
            "experience_levels")
    return {k: v for k, v in zip(keys, row) if v is not None}


@router.get("/deck/jobs")
def job_deck(
    request: Request,
    limit: int = Query(20, ge=1, le=50),
    user: CurrentUser = Depends(get_current_user),
) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select resume_text, embedding::text, years_experience "
                "from public.seeker_profiles where user_id = %s",
                (user.id,),
            )
            row = cur.fetchone()
            if not row or not row[1]:
                raise HTTPException(400, "Upload a resume before requesting a deck")
            resume_text, embedding_text, years = row
            survey = _load_survey(cur, user.id)

        cards = rank_jobs_for_resume(
            conn,
            request.app.state.scorer,
            resume_text,
            parse_vector(embedding_text),
            survey,
            user.id,
            limit,
            float(years) if years is not None else None,
        )

    for card in cards:
        card["id"] = str(card["id"])
        card.pop("employer_id", None)
        card["normalized_salary"] = (
            float(card["normalized_salary"]) if card["normalized_salary"] is not None else None
        )
        # Cards only need a readable excerpt; shipping 4,000 characters per card to the
        # browser for a 20-card deck is 80 KB of mostly unread text.
        card["description"] = (card["description"] or "")[:1200]
    return cards


@router.get("/deck/candidates")
def candidate_deck(
    request: Request,
    job_id: str = Query(...),
    limit: int = Query(20, ge=1, le=50),
    user: CurrentUser = Depends(get_current_user),
) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select j.title, j.company_name, j.description, j.embedding::text
                from public.jobs j
                join public.employer_profiles e on e.id = j.employer_id
                where j.id = %s and e.user_id = %s
                """,
                (job_id, user.id),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Job not found, or it is not yours")
            title, company, description, embedding_text = row
            if not embedding_text:
                raise HTTPException(409, "This posting has no embedding yet")

        from ml.encoder import job_text

        cards = rank_candidates_for_job(
            conn,
            request.app.state.scorer,
            job_text(title, company, description),
            parse_vector(embedding_text),
            user.id,
            limit,
        )

    for card in cards:
        card["id"] = str(card["id"])
        card.pop("user_id", None)
        card["years_experience"] = (
            float(card["years_experience"]) if card["years_experience"] is not None else None
        )
    return cards


@router.post("/swipe", response_model=SwipeOut)
def swipe(payload: SwipeIn, user: CurrentUser = Depends(get_current_user)) -> SwipeOut:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("select role::text from public.profiles where id = %s", (user.id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(400, "Complete your profile first")
        actor_role = row[0]

        cur.execute(
            """
            insert into public.swipes
                (actor_user_id, actor_role, target_type, target_id, direction, score_at_swipe)
            values (%s, %s, %s, %s, %s, %s)
            on conflict (actor_user_id, target_type, target_id) do update set
                direction = excluded.direction,
                score_at_swipe = excluded.score_at_swipe
            """,
            (user.id, actor_role, payload.target_type, payload.target_id,
             payload.direction, payload.score_at_swipe),
        )

        if payload.direction != "like":
            conn.commit()
            return SwipeOut(recorded=True)

        result = (
            _seeker_liked_job(cur, user.id, payload.target_id)
            if payload.target_type == "job"
            else _employer_liked_seeker(cur, user.id, payload.target_id)
        )
        conn.commit()
        return result


def _seeker_liked_job(cur, seeker_user_id: str, job_id: str) -> SwipeOut:
    """A seeker liked a posting.

    Sourced listings (LinkedIn seed, USAJOBS) have no employer account behind them, so
    waiting for a mutual like would mean no seeker ever matches anything. They are public
    postings with real apply URLs, so a like is enough: match immediately.

    Employer-posted jobs are the two-sided case and require the employer to have liked
    this candidate back.
    """
    cur.execute(
        "select j.application_url, j.employer_id, e.user_id "
        "from public.jobs j left join public.employer_profiles e on e.id = j.employer_id "
        "where j.id = %s",
        (job_id,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(404, "Job not found")
    application_url, employer_id, employer_user_id = row

    if employer_id is not None:
        cur.execute(
            """
            select 1 from public.swipes s
            join public.seeker_profiles p on p.id = s.target_id
            where s.actor_user_id = %s and s.target_type = 'seeker'
              and s.direction = 'like' and p.user_id = %s
            """,
            (employer_user_id, seeker_user_id),
        )
        if cur.fetchone() is None:
            return SwipeOut(recorded=True, matched=False)

    cur.execute(
        "insert into public.matches (seeker_user_id, job_id, employer_user_id, application_url) "
        "values (%s, %s, %s, %s) on conflict (seeker_user_id, job_id) do nothing "
        "returning id",
        (seeker_user_id, job_id, employer_user_id, application_url),
    )
    row = cur.fetchone()
    if row is None:  # already matched
        cur.execute(
            "select id from public.matches where seeker_user_id = %s and job_id = %s",
            (seeker_user_id, job_id),
        )
        row = cur.fetchone()

    return SwipeOut(recorded=True, matched=True, match_id=str(row[0]),
                    application_url=application_url)


def _employer_liked_seeker(cur, employer_user_id: str, seeker_profile_id: str) -> SwipeOut:
    """An employer liked a candidate; match if that candidate liked one of their jobs.

    Demo profiles loaded from Resume.csv have no auth user, so they can be swiped on but
    can never match -- there is nobody on the other side to have liked back.
    """
    cur.execute(
        "select user_id, is_demo from public.seeker_profiles where id = %s",
        (seeker_profile_id,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(404, "Candidate not found")
    seeker_user_id, is_demo = row
    if is_demo or seeker_user_id is None:
        return SwipeOut(recorded=True, matched=False)

    cur.execute(
        """
        select s.target_id, j.application_url
        from public.swipes s
        join public.jobs j on j.id = s.target_id
        join public.employer_profiles e on e.id = j.employer_id
        where s.actor_user_id = %s and s.target_type = 'job' and s.direction = 'like'
          and e.user_id = %s
        order by s.created_at desc
        limit 1
        """,
        (seeker_user_id, employer_user_id),
    )
    row = cur.fetchone()
    if not row:
        return SwipeOut(recorded=True, matched=False)
    job_id, application_url = row

    cur.execute(
        "insert into public.matches (seeker_user_id, job_id, employer_user_id, application_url) "
        "values (%s, %s, %s, %s) on conflict (seeker_user_id, job_id) do nothing returning id",
        (seeker_user_id, job_id, employer_user_id, application_url),
    )
    row = cur.fetchone()
    if row is None:
        cur.execute(
            "select id from public.matches where seeker_user_id = %s and job_id = %s",
            (seeker_user_id, job_id),
        )
        row = cur.fetchone()

    return SwipeOut(recorded=True, matched=True, match_id=str(row[0]),
                    application_url=application_url)


@router.get("/matches")
def list_matches(user: CurrentUser = Depends(get_current_user)) -> list[dict]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select m.id, m.job_id, j.title as job_title, j.company_name, m.application_url,
                   j.source::text, m.created_at,
                   case when m.seeker_user_id = %(uid)s then e.company_name
                        else sp.display_name end as counterpart_name
            from public.matches m
            join public.jobs j on j.id = m.job_id
            left join public.employer_profiles e on e.id = j.employer_id
            left join public.seeker_profiles sp on sp.user_id = m.seeker_user_id
            where m.seeker_user_id = %(uid)s or m.employer_user_id = %(uid)s
            order by m.created_at desc
            """,
            {"uid": user.id},
        )
        columns = [c.name for c in cur.description]
        rows = [dict(zip(columns, r)) for r in cur.fetchall()]

    for row in rows:
        row["id"] = str(row["id"])
        row["job_id"] = str(row["job_id"])
        row["created_at"] = row["created_at"].isoformat()
    return rows
