"""Audit, clean and split the Supabase data into an `analytics` schema.

The app's own tables (`public.jobs`, `public.seeker_profiles`) are never modified. Every
cleaning decision lands as a flag or a cleaned column in `analytics`, keyed by the source
row's id, so the raw value is always one join away and the whole schema can be dropped
and rebuilt from scratch:

    python analytics/prepare.py            # rebuild everything, print the audit
    python analytics/prepare.py --dry-run  # print the audit, write nothing

What it builds:

    analytics.job_quality       one row per sourced job: cleaned columns, flags, split
    analytics.resume_quality    one row per demo resume: cleaned columns, flags, split
    analytics.jobs_clean        view: jobs joined to job_quality, minus embeddings
    analytics.resumes_clean     view: demo resumes joined to resume_quality, minus text
    analytics.audit_checks      every check run, rows affected, and the action taken
    analytics.data_dictionary   every column the views expose, with measured null rates
    analytics.split_summary     train/test counts per stratum

Scope: only `source <> 'employer'` jobs and `is_demo` resumes. Real users' resumes and
employer-posted jobs are personal or customer data and stay out of analysis entirely.

The reserved test split (20%, `split = 'test'`) is stratified -- by experience level for
jobs, by category for resumes -- and group-safe: rows whose text is byte-identical share
a split, so a duplicate can never sit on both sides. Assignment is a pure function of the
row id and text (md5 with a fixed salt), so rebuilding never reshuffles it. Exploratory
analysis and the dashboard read `split = 'train'` only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.db import connect  # noqa: E402

TEST_FRACTION = 0.20
# Changing the salt reshuffles the holdout. Bump it only if the test set has been
# compromised (looked at, tuned against) and a fresh one is genuinely needed.
SPLIT_SALT = "job-tinder-holdout-v1"

# Plausible annual base pay. Below the floor is almost always a mislabelled pay period
# (a monthly or hourly figure stored as yearly, or a literal 0); above the ceiling is an
# hourly rate multiplied by 2,080 hours after it was already annualised.
SALARY_FLOOR = 15_000
SALARY_CEILING = 1_000_000

DESC_CAP = 4_000     # data/load_postings.py --desc-chars default
RESUME_CAP = 8_000   # data/load_resumes.py --max-chars default

# LinkedIn metro-area labels that data/load_postings.py's ", XX" parser can't place.
# Metros that straddle a state line are assigned the state of their principal city.
METRO_STATES = {
    "San Francisco Bay Area": "CA", "Los Angeles Metropolitan Area": "CA",
    "Dallas-Fort Worth Metroplex": "TX", "Atlanta Metropolitan Area": "GA",
    "Greater Chicago Area": "IL", "Greater Houston": "TX", "Greater Boston": "MA",
    "Nashville Metropolitan Area": "TN", "Greater Phoenix Area": "AZ",
    "Salt Lake City Metropolitan Area": "UT", "Greater Orlando": "FL",
    "Greater Minneapolis-St. Paul Area": "MN", "Raleigh-Durham-Chapel Hill Area": "NC",
    "Denver Metropolitan Area": "CO", "Miami-Fort Lauderdale Area": "FL",
    "Greater Philadelphia": "PA", "Greater Seattle Area": "WA",
    "Cincinnati Metropolitan Area": "OH", "Greater St. Louis": "MO",
    "Detroit Metropolitan Area": "MI", "San Diego Metropolitan Area": "CA",
    "Charlotte Metro": "NC", "Greater Cleveland": "OH", "Greater Sacramento": "CA",
    "Greater Tampa Bay Area": "FL", "Omaha Metropolitan Area": "NE",
    "Las Vegas Metropolitan Area": "NV", "Grand Rapids Metropolitan Area": "MI",
    "Utica-Rome Area": "NY", "Greater Pittsburgh Region": "PA",
    "Gainesville Metropolitan Area": "FL", "Memphis Metropolitan Area": "TN",
    "Greensboro--Winston-Salem--High Point Area": "NC",
    "Louisville Metropolitan Area": "KY", "Topeka Metropolitan Area": "KS",
    "Knoxville Metropolitan Area": "TN", "Greater Milwaukee": "WI",
    "Baton Rouge Metropolitan Area": "LA", "Greater Corpus Christi Area": "TX",
    "Appleton-Oshkosh-Neenah Area": "WI", "Greater Richmond Region": "VA",
    "Eau Claire-Menomonie Area": "WI", "Little Rock Metropolitan Area": "AR",
}
# Locations that name the whole country or region: genuinely no state, not a parse miss.
NATIONWIDE = ("United States", "NAMER")


def split_sql(stratum: str, group_key: str, source: str) -> str:
    """Stratified, group-safe holdout assignment as a SQL expression over `source`.

    Within each stratum, groups (identical text) are ordered by a salted hash and the
    first TEST_FRACTION of rows go to test. A group whose first row makes the cut goes
    to test whole, so a stratum can land a row or two over the target.
    """
    return f"""
    with keyed as (
        select id, {stratum} as stratum,
               md5('{SPLIT_SALT}' || {group_key}) as g
        from {source}
    ), ranked as (
        select id, stratum, g,
               row_number() over (partition by stratum order by g, id) as rn,
               count(*) over (partition by stratum) as n
        from keyed
    ), groups as (
        select g, min(rn) as first_rn, max(n) as n from ranked group by g
    )
    select r.id,
           case when gr.first_rn <= ceil(gr.n * {TEST_FRACTION}) then 'test' else 'train' end
    from ranked r join groups gr using (g)
    """


DDL = f"""
create schema if not exists analytics;

drop view if exists analytics.jobs_clean;
drop view if exists analytics.resumes_clean;
drop table if exists analytics.job_quality, analytics.resume_quality,
    analytics.audit_checks, analytics.data_dictionary, analytics.split_summary,
    analytics.metro_states;

create table analytics.metro_states (location text primary key, state text not null);

create table analytics.job_quality (
    job_id                 uuid primary key,
    company_name_clean     text,
    state_clean            text,
    is_nationwide          boolean not null,
    salary_annual_clean    numeric,
    salary_missing         boolean not null,
    salary_out_of_range    boolean not null,
    description_chars      integer not null,
    description_truncated  boolean not null,
    description_dup_group  integer not null,
    split                  text    not null check (split in ('train', 'test'))
);

create table analytics.resume_quality (
    seeker_profile_id      uuid primary key,
    category               text    not null,
    years_experience_est   numeric,
    resume_chars           integer not null,
    resume_truncated       boolean not null,
    resume_dup_group       integer not null,
    split                  text    not null check (split in ('train', 'test'))
);

create table analytics.audit_checks (
    check_id    text primary key,
    entity      text    not null,
    description text    not null,
    n_affected  integer not null,
    n_total     integer not null,
    severity    text    not null,
    action      text    not null
);

create table analytics.data_dictionary (
    view_name   text not null,
    column_name text not null,
    data_type   text not null,
    meaning     text not null,
    cleaning    text not null,
    null_pct    numeric,
    primary key (view_name, column_name)
);

create table analytics.split_summary (
    entity  text not null,
    stratum text not null,
    n_train integer not null,
    n_test  integer not null,
    primary key (entity, stratum)
);
"""

POPULATE = f"""
insert into analytics.job_quality
select j.id,
       nullif(trim(j.company_name), ''),
       coalesce(j.state, m.state),
       j.state is null and m.state is null and j.location in {NATIONWIDE!r},
       case when j.normalized_salary between {SALARY_FLOOR} and {SALARY_CEILING}
            then round(j.normalized_salary) end,
       j.normalized_salary is null,
       j.normalized_salary is not null
           and j.normalized_salary not between {SALARY_FLOOR} and {SALARY_CEILING},
       length(j.description),
       length(j.description) >= {DESC_CAP},
       count(*) over (partition by md5(j.description))::int,
       s.split
from public.jobs j
left join analytics.metro_states m on m.location = j.location
join ({split_sql("j.formatted_experience_level", "md5(j.description)",
                 "public.jobs j where j.source <> 'employer'")}) s(id, split) on s.id = j.id
where j.source <> 'employer';

insert into analytics.resume_quality
select p.id,
       p.predicted_category,
       p.years_experience,
       length(p.resume_text),
       length(p.resume_text) >= {RESUME_CAP},
       count(*) over (partition by md5(p.resume_text))::int,
       s.split
from public.seeker_profiles p
join ({split_sql("p.predicted_category", "md5(p.resume_text)",
                 "public.seeker_profiles p where p.is_demo")}) s(id, split) on s.id = p.id
where p.is_demo;

create view analytics.jobs_clean as
select j.id, j.title, q.company_name_clean as company_name, j.location, q.state_clean as state,
       q.is_nationwide, j.remote_allowed, j.formatted_work_type as work_type,
       j.formatted_experience_level as experience_level, j.pay_period,
       j.normalized_salary as salary_raw, q.salary_annual_clean as salary_annual,
       q.salary_missing, q.salary_out_of_range, q.description_chars,
       q.description_truncated, q.description_dup_group, q.split
from public.jobs j join analytics.job_quality q on q.job_id = j.id;

create view analytics.resumes_clean as
select p.id, q.category, q.years_experience_est, q.resume_chars, q.resume_truncated,
       q.resume_dup_group, q.split
from public.seeker_profiles p join analytics.resume_quality q on q.seeker_profile_id = p.id;

insert into analytics.split_summary
select 'jobs', experience_level, count(*) filter (where split = 'train'),
       count(*) filter (where split = 'test')
from analytics.jobs_clean group by experience_level
union all
select 'resumes', category, count(*) filter (where split = 'train'),
       count(*) filter (where split = 'test')
from analytics.resumes_clean group by category;
"""

# (check_id, entity, description, SQL for n_affected, severity, action)
CHECKS = [
    ("salary_missing", "jobs", "No salary posted",
     "select count(*) from analytics.job_quality where salary_missing",
     "high", "Kept; salary_annual is null. Salary analysis covers posted salaries only."),
    ("salary_out_of_range", "jobs",
     f"Salary outside ${SALARY_FLOOR:,}-${SALARY_CEILING:,}/yr (mislabelled pay period)",
     "select count(*) from analytics.job_quality where salary_out_of_range",
     "medium", "Kept; salary_annual set to null, raw value preserved in salary_raw."),
    ("posted_at_missing", "jobs", "posted_at empty (loader never mapped listed_time)",
     "select count(*) from public.jobs where source <> 'employer' and posted_at is null",
     "high", "Column excluded from the clean view; no time-based analysis possible."),
    ("state_unparsed", "jobs", "State missing but location is a known metro area",
     "select count(*) from public.jobs j join analytics.metro_states m using (location) "
     "where j.state is null and j.source <> 'employer'",
     "low", "Recovered via analytics.metro_states (principal city's state)."),
    ("state_nationwide", "jobs", "Location is the whole country (no state exists)",
     "select count(*) from analytics.job_quality where is_nationwide",
     "low", "Kept; flagged is_nationwide, excluded from per-state counts."),
    ("state_still_missing", "jobs", "State missing after metro recovery, not nationwide",
     "select count(*) from analytics.job_quality where state_clean is null and not is_nationwide",
     "low", "Kept; state is null."),
    ("company_blank", "jobs", "Blank company name",
     "select count(*) from analytics.job_quality where company_name_clean is null",
     "low", "Kept; company_name set to null."),
    ("description_truncated", "jobs", f"Description cut at the {DESC_CAP:,}-char load cap",
     "select count(*) from analytics.job_quality where description_truncated",
     "medium", "Kept; flagged. Length statistics are censored at the cap."),
    ("description_duplicate", "jobs", "Description byte-identical to another posting",
     "select count(*) from analytics.job_quality where description_dup_group > 1",
     "medium", "Kept (different titles); duplicates forced into the same split."),
    ("resume_truncated", "resumes", f"Resume cut at the {RESUME_CAP:,}-char load cap",
     "select count(*) from analytics.resume_quality where resume_truncated",
     "low", "Kept; flagged."),
    ("resume_duplicate", "resumes", "Resume byte-identical to another resume",
     "select count(*) from analytics.resume_quality where resume_dup_group > 1",
     "low", "Kept; duplicates forced into the same split."),
    ("yoe_heuristic", "resumes", "Years of experience is a regex estimate, not reported",
     "select count(*) from analytics.resume_quality where years_experience_est is not null",
     "medium", "Kept; treat as approximate."),
    ("yoe_missing", "resumes", "No years-of-experience estimate",
     "select count(*) from analytics.resume_quality where years_experience_est is null",
     "low", "Kept; null."),
]
TOTALS = {
    "jobs": "select count(*) from analytics.job_quality",
    "resumes": "select count(*) from analytics.resume_quality",
}

# (view, column, meaning, cleaning). Types and null rates are read from the database.
DICTIONARY = [
    ("jobs_clean", "id", "Job id (public.jobs.id)", "None"),
    ("jobs_clean", "title", "Posting title as listed on LinkedIn", "None"),
    ("jobs_clean", "company_name", "Hiring company", "Blank strings set to null"),
    ("jobs_clean", "location", "Free-text location as listed", "None"),
    ("jobs_clean", "state", "US state code", "Metro-area labels mapped to a state"),
    ("jobs_clean", "is_nationwide", "Location is the whole US, so no state applies", "Derived"),
    ("jobs_clean", "remote_allowed", "Posting allows remote work", "None"),
    ("jobs_clean", "work_type", "Full-time, Contract, Part-time, Internship, Temporary",
     "Volunteer/Other dropped at load"),
    ("jobs_clean", "experience_level", "LinkedIn seniority band (6 levels)",
     "Rows without one dropped at load"),
    ("jobs_clean", "pay_period", "Period the posted salary was quoted in", "None"),
    ("jobs_clean", "salary_raw", "LinkedIn's annualised salary, unvalidated", "None"),
    ("jobs_clean", "salary_annual", "Annualised salary, USD",
     f"Null outside ${SALARY_FLOOR:,}-${SALARY_CEILING:,}"),
    ("jobs_clean", "salary_missing", "No salary posted", "Derived"),
    ("jobs_clean", "salary_out_of_range", "Posted salary failed the range check", "Derived"),
    ("jobs_clean", "description_chars", "Description length in characters",
     f"Censored at {DESC_CAP:,}; rows under 700 dropped at load"),
    ("jobs_clean", "description_truncated", f"Description hit the {DESC_CAP:,}-char cap",
     "Derived"),
    ("jobs_clean", "description_dup_group", "Postings sharing this exact description",
     "Derived"),
    ("jobs_clean", "split", "train, or test (reserved holdout)", "Stratified by experience_level"),
    ("resumes_clean", "id", "Demo profile id (public.seeker_profiles.id)", "None"),
    ("resumes_clean", "category", "Resume.csv job category (24 values)", "None"),
    ("resumes_clean", "years_experience_est", "Years of experience estimated from text",
     "Regex heuristic at load"),
    ("resumes_clean", "resume_chars", "Resume length in characters",
     f"Censored at {RESUME_CAP:,}"),
    ("resumes_clean", "resume_truncated", f"Resume hit the {RESUME_CAP:,}-char cap", "Derived"),
    ("resumes_clean", "resume_dup_group", "Resumes sharing this exact text", "Derived"),
    ("resumes_clean", "split", "train, or test (reserved holdout)", "Stratified by category"),
]


def build(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
        cur.executemany("insert into analytics.metro_states values (%s, %s)",
                        list(METRO_STATES.items()))
        cur.execute(POPULATE)

        totals = {k: cur.execute(v).fetchone()[0] for k, v in TOTALS.items()}
        for check_id, entity, desc, sql, severity, action in CHECKS:
            n = cur.execute(sql).fetchone()[0]
            cur.execute("insert into analytics.audit_checks values (%s,%s,%s,%s,%s,%s,%s)",
                        (check_id, entity, desc, n, totals[entity], severity, action))

        for view, column, meaning, cleaning in DICTIONARY:
            dtype = cur.execute(
                "select data_type from information_schema.columns "
                "where table_schema = 'analytics' and table_name = %s and column_name = %s",
                (view, column)).fetchone()[0]
            # Identifiers come from the fixed DICTIONARY above, never from input.
            null_pct = cur.execute(
                f"select round(100.0 * avg(({column} is null)::int), 1) from analytics.{view}"
            ).fetchone()[0]
            cur.execute("insert into analytics.data_dictionary values (%s,%s,%s,%s,%s,%s)",
                        (view, column, dtype, meaning, cleaning, null_pct))
    conn.commit()


def report(conn) -> None:
    rows = conn.execute(
        "select check_id, n_affected, n_total, severity from analytics.audit_checks "
        "order by entity, check_id").fetchall()
    print(f"{'check':<24}{'affected':>10}{'of':>8}{'pct':>8}  severity")
    for check_id, n, total, severity in rows:
        print(f"{check_id:<24}{n:>10,}{total:>8,}{100 * n / total:>7.1f}%  {severity}")
    print()
    for entity, train, test in conn.execute(
            "select entity, sum(n_train), sum(n_test) from analytics.split_summary "
            "group by entity order by entity").fetchall():
        print(f"{entity:<8} train {train:>6,}  test {test:>5,}  "
              f"({100 * test / (train + test):.1f}% reserved)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="build inside a transaction, print the audit, roll back")
    args = parser.parse_args()
    with connect() as conn:
        if args.dry_run:
            # build() commits; swap in a no-op so the whole run stays one transaction.
            conn.commit = lambda: None
            build(conn)
            report(conn)
            conn.rollback()
            print("\n(dry run: nothing written)")
        else:
            build(conn)
            report(conn)


if __name__ == "__main__":
    main()
