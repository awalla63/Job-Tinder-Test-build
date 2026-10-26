"""Profile creation, resume upload, survey, and employer job posting."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from services.api.auth import CurrentUser, get_current_user
from services.api.database import get_connection
from services.api.resume_parse import UnsupportedResume, extract_text
from services.api.schemas import CompleteProfileIn, JobIn, SurveyIn

log = logging.getLogger(__name__)
router = APIRouter(tags=["onboarding"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _vector_literal(vector) -> str:
    return "[" + ",".join(f"{float(v):.6f}" for v in vector) + "]"


@router.get("/me")
def me(user: CurrentUser = Depends(get_current_user)) -> dict:
    """Everything the client needs to decide which onboarding step to show."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("select role::text, display_name from public.profiles where id = %s",
                    (user.id,))
        row = cur.fetchone()
        if not row:
            return {"onboarded": False, "role": None}
        role, display_name = row

        cur.execute("select 1 from public.surveys where user_id = %s", (user.id,))
        has_survey = cur.fetchone() is not None

        has_resume = False
        company_name = None
        job_count = 0
        if role == "seeker":
            cur.execute(
                "select resume_text <> '' from public.seeker_profiles where user_id = %s",
                (user.id,),
            )
            result = cur.fetchone()
            has_resume = bool(result and result[0])
        else:
            cur.execute(
                "select e.company_name, count(j.id) "
                "from public.employer_profiles e "
                "left join public.jobs j on j.employer_id = e.id "
                "where e.user_id = %s group by e.company_name",
                (user.id,),
            )
            result = cur.fetchone()
            if result:
                company_name, job_count = result[0], int(result[1])

    return {
        "onboarded": True,
        "role": role,
        "display_name": display_name,
        "has_survey": has_survey,
        "has_resume": has_resume,
        "company_name": company_name,
        "job_count": job_count,
    }


@router.post("/auth/complete-profile")
def complete_profile(
    payload: CompleteProfileIn, user: CurrentUser = Depends(get_current_user)
) -> dict:
    if payload.role == "employer" and not (payload.company_name or "").strip():
        raise HTTPException(400, "company_name is required for employers")

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "insert into public.profiles (id, role, display_name) values (%s, %s, %s) "
            "on conflict (id) do update set role = excluded.role, "
            "display_name = excluded.display_name",
            (user.id, payload.role, payload.display_name),
        )
        if payload.role == "seeker":
            cur.execute(
                "insert into public.seeker_profiles (user_id, display_name) values (%s, %s) "
                "on conflict (user_id) do update set display_name = excluded.display_name",
                (user.id, payload.display_name),
            )
        else:
            cur.execute(
                "insert into public.employer_profiles (user_id, company_name) "
                "values (%s, %s) on conflict (user_id) do update set "
                "company_name = excluded.company_name",
                (user.id, payload.company_name.strip()),
            )
        conn.commit()
    return {"ok": True, "role": payload.role}


@router.post("/seeker/resume")
async def upload_resume(
    request: Request,
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Resume must be under 10 MB")

    try:
        text = extract_text(payload, file.filename or "")
    except UnsupportedResume as exc:
        raise HTTPException(400, str(exc)) from exc

    encoder = request.app.state.encoder
    embedding = encoder.encode_resumes([text])[0]

    from ml.features import resume_years

    years = resume_years(text)
    years = None if years != years else float(years)  # NaN -> None

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into public.seeker_profiles
                (user_id, resume_text, resume_path, years_experience, embedding, updated_at)
            values (%s, %s, %s, %s, %s::halfvec, now())
            on conflict (user_id) do update set
                resume_text = excluded.resume_text,
                resume_path = excluded.resume_path,
                years_experience = excluded.years_experience,
                embedding = excluded.embedding,
                updated_at = now()
            """,
            (user.id, text, file.filename, years, _vector_literal(embedding)),
        )
        conn.commit()

    return {
        "ok": True,
        "characters": len(text),
        "years_experience": years,
        "preview": text[:300],
    }


@router.post("/survey")
def save_survey(payload: SurveyIn, user: CurrentUser = Depends(get_current_user)) -> dict:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("select role::text from public.profiles where id = %s", (user.id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(400, "Complete your profile first")
        role = row[0]

        cur.execute(
            """
            insert into public.surveys (user_id, role, answers, min_salary, remote_pref,
                                        work_types, states, categories, experience_levels,
                                        updated_at)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            on conflict (user_id) do update set
                answers = excluded.answers,
                min_salary = excluded.min_salary,
                remote_pref = excluded.remote_pref,
                work_types = excluded.work_types,
                states = excluded.states,
                categories = excluded.categories,
                experience_levels = excluded.experience_levels,
                updated_at = now()
            """,
            (
                user.id, role, __import__("json").dumps(payload.answers),
                payload.min_salary,
                None if payload.remote_pref == "any" else payload.remote_pref,
                payload.work_types, payload.states, payload.categories,
                payload.experience_levels,
            ),
        )
        conn.commit()
    return {"ok": True}


@router.post("/employer/jobs")
def create_job(
    payload: JobIn, request: Request, user: CurrentUser = Depends(get_current_user)
) -> dict:
    encoder = request.app.state.encoder

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "select id, company_name from public.employer_profiles where user_id = %s",
            (user.id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(400, "Create an employer profile first")
        employer_id, company_name = row

        company = payload.company_name or company_name
        from ml.encoder import job_text

        embedding = encoder.encode_jobs([job_text(payload.title, company, payload.description)])[0]

        cur.execute(
            """
            insert into public.jobs
                (source, employer_id, company_name, title, description, location, state,
                 normalized_salary, pay_period, formatted_work_type,
                 formatted_experience_level, remote_allowed, application_url,
                 posted_at, embedding)
            values ('employer', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    now(), %s::halfvec)
            returning id
            """,
            (
                employer_id, company, payload.title, payload.description, payload.location,
                payload.state, payload.normalized_salary, payload.pay_period,
                payload.formatted_work_type, payload.formatted_experience_level,
                payload.remote_allowed, payload.application_url,
                _vector_literal(embedding),
            ),
        )
        job_id = cur.fetchone()[0]
        conn.commit()

    return {"ok": True, "job_id": str(job_id)}


@router.get("/employer/jobs")
def list_jobs(user: CurrentUser = Depends(get_current_user)) -> list[dict]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select j.id, j.title, j.company_name, j.location, j.remote_allowed,
                   j.created_at,
                   (select count(*) from public.swipes s
                     where s.target_type = 'job' and s.target_id = j.id
                       and s.direction = 'like') as likes
            from public.jobs j
            join public.employer_profiles e on e.id = j.employer_id
            where e.user_id = %s
            order by j.created_at desc
            """,
            (user.id,),
        )
        columns = [c.name for c in cur.description]
        rows = [dict(zip(columns, r)) for r in cur.fetchall()]

    for row in rows:
        row["id"] = str(row["id"])
        row["created_at"] = row["created_at"].isoformat()
        row["likes"] = int(row["likes"])
    return rows
