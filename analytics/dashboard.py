"""Job Tinder data dashboard (Streamlit).

Reads the `analytics` schema built by analytics/prepare.py. Every query is an aggregate
over the train split, so no posting text, resume text or row-level personal data ever
reaches the page, and the reserved test rows are only ever counted, never plotted.

    streamlit run analytics/dashboard.py        # locally, reads DATABASE_URL from .env
    modal deploy analytics/modal_dashboard.py   # hosted
"""
from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import psycopg
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # the Modal image passes DATABASE_URL in through the secret
    pass

EXPERIENCE_ORDER = ["Internship", "Entry level", "Associate", "Mid-Senior level",
                    "Director", "Executive"]
ACCENT = "#2563eb"
MUTED = "#9ca3af"

st.set_page_config(page_title="Job Tinder data", layout="wide")


@st.cache_resource
def connection() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    # The URL is the service role's, so enforce read-only at the session level: nothing
    # on this page should ever be able to write.
    conn.execute("set session characteristics as transaction read only")
    return conn


@st.cache_data(ttl=600)
def query(sql: str) -> pd.DataFrame:
    with connection().cursor() as cur:
        cur.execute(sql)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def num(df: pd.DataFrame) -> pd.DataFrame:
    """Postgres numerics arrive as Decimal; Altair wants floats."""
    for c in df.columns:
        if df[c].dtype == object and len(df) and hasattr(df[c].iloc[0], "as_integer_ratio"):
            df[c] = df[c].astype(float)
    return df


st.title("Job Tinder — data audit")
st.caption("Source: Supabase `analytics` schema (analytics/prepare.py). "
           "Charts use the train split only; the reserved test split is counted, never plotted.")

# --- Headline numbers -------------------------------------------------------
split = num(query("""
    select entity, sum(n_train) as train, sum(n_test) as test
    from analytics.split_summary group by entity order by entity
"""))
jobs_row = split[split.entity == "jobs"].iloc[0]
res_row = split[split.entity == "resumes"].iloc[0]
cov = num(query("""
    select round(100.0 * avg((salary_annual is not null)::int), 1) as salary_pct,
           round(100.0 * avg(remote_allowed::int), 1) as remote_pct
    from analytics.jobs_clean where split = 'train'
"""))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jobs (train / test)", f"{int(jobs_row.train):,} / {int(jobs_row.test):,}")
c2.metric("Demo resumes (train / test)", f"{int(res_row.train):,} / {int(res_row.test):,}")
c3.metric("Jobs with a usable salary", f"{cov.salary_pct[0]:.1f}%")
c4.metric("Remote-allowed jobs", f"{cov.remote_pct[0]:.1f}%")

tab_plots, tab_audit, tab_dict, tab_split = st.tabs(
    ["Plots", "Audit checks", "Data dictionary", "Reserved test split"])

with tab_plots:
    # 1. Data quality: share of jobs affected by each check.
    st.subheader("1. Two fields carry most of the quality problems")
    audit = num(query("""
        select description, 100.0 * n_affected / n_total as pct
        from analytics.audit_checks where entity = 'jobs' order by pct desc
    """))
    audit["top"] = audit.pct >= 50
    st.altair_chart(
        alt.Chart(audit).mark_bar().encode(
            x=alt.X("pct:Q", title="Share of jobs affected (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("description:N", sort="-x", title=None),
            color=alt.condition("datum.top", alt.value(ACCENT), alt.value(MUTED)),
            tooltip=["description", alt.Tooltip("pct:Q", format=".1f")],
        ).properties(height=300),
        use_container_width=True)
    st.markdown("`posted_at` is empty for every posting and two thirds post no salary. "
                "Everything else affects under half the rows and is flagged, not dropped.")

    # 2. Salary by experience level: median with interquartile range.
    st.subheader("2. Median posted salary rises 4x from internship to executive")
    sal = num(query("""
        select experience_level, count(*) as n, count(salary_annual) as n_salary,
               percentile_cont(0.25) within group (order by salary_annual) as p25,
               percentile_cont(0.5)  within group (order by salary_annual) as median,
               percentile_cont(0.75) within group (order by salary_annual) as p75
        from analytics.jobs_clean where split = 'train' group by experience_level
    """))
    sal["coverage"] = (100 * sal.n_salary / sal.n).round(0)
    base = alt.Chart(sal).encode(
        x=alt.X("experience_level:N", sort=EXPERIENCE_ORDER, title=None))
    st.altair_chart(
        (base.mark_rule(color=MUTED, strokeWidth=2).encode(
            y=alt.Y("p25:Q", title="Annual salary (USD)", axis=alt.Axis(format="$,.0f")),
            y2="p75:Q")
         + base.mark_point(filled=True, size=90, color=ACCENT).encode(
            y="median:Q",
            tooltip=["experience_level", alt.Tooltip("median:Q", format="$,.0f"),
                     alt.Tooltip("p25:Q", format="$,.0f"), alt.Tooltip("p75:Q", format="$,.0f"),
                     alt.Tooltip("coverage:Q", title="% with salary")])
         ).properties(height=320),
        use_container_width=True)
    st.markdown("Dot = median, line = 25th–75th percentile, over postings that list a salary. "
                f"Only {sal.coverage.min():.0f}–{sal.coverage.max():.0f}% of postings per "
                "level list one, so these describe the disclosing subset, not the market.")

    # 3. Seniority mismatch between the two sides.
    st.subheader("3. Jobs skew junior, demo resumes skew senior")
    mix = num(query("""
        with jobs as (
            select case when experience_level in ('Internship', 'Entry level') then '0-2 yrs'
                        when experience_level = 'Associate' then '3-5 yrs'
                        when experience_level = 'Mid-Senior level' then '6-15 yrs'
                        else '16+ yrs' end as band, count(*) as n
            from analytics.jobs_clean where split = 'train' group by 1
        ), resumes as (
            select case when years_experience_est <= 2 then '0-2 yrs'
                        when years_experience_est <= 5 then '3-5 yrs'
                        when years_experience_est <= 15 then '6-15 yrs'
                        else '16+ yrs' end as band, count(*) as n
            from analytics.resumes_clean
            where split = 'train' and years_experience_est is not null group by 1
        )
        select 'Jobs (experience level)' as side, band, 100.0 * n / sum(n) over () as pct from jobs
        union all
        select 'Resumes (estimated years)', band, 100.0 * n / sum(n) over () from resumes
    """))
    st.altair_chart(
        alt.Chart(mix).mark_bar().encode(
            x=alt.X("band:N", sort=["0-2 yrs", "3-5 yrs", "6-15 yrs", "16+ yrs"], title=None),
            xOffset="side:N",
            y=alt.Y("pct:Q", title="Share of side (%)"),
            color=alt.Color("side:N", scale=alt.Scale(range=[ACCENT, MUTED]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["side", "band", alt.Tooltip("pct:Q", format=".1f")],
        ).properties(height=320),
        use_container_width=True)
    st.markdown("Levels mapped to years as Internship/Entry = 0–2, Associate = 3–5, "
                "Mid-Senior = 6–15, Director/Executive = 16+ (an approximation). "
                "Ranking quality for entry-level seekers is barely tested by this corpus.")

    # 4. Resume category balance.
    st.subheader("4. Three resume categories are too small for per-category metrics")
    cats = num(query("""
        select category, count(*) as n from analytics.resumes_clean
        where split = 'train' group by category
    """))
    cats["small"] = cats.n < 60
    st.altair_chart(
        alt.Chart(cats).mark_bar().encode(
            x=alt.X("n:Q", title="Training resumes"),
            y=alt.Y("category:N", sort="-x", title=None),
            color=alt.condition("datum.small", alt.value(ACCENT), alt.value(MUTED)),
            tooltip=["category", "n"],
        ).properties(height=520),
        use_container_width=True)

    # 5. Description length: the load-time cap is visible as a spike.
    st.subheader("5. 45% of job descriptions are cut off at the 4,000-character cap")
    lengths = num(query("""
        select least(width_bucket(description_chars, 700, 4000, 11), 12) as bucket, count(*) as n
        from analytics.jobs_clean where split = 'train' group by 1 order by 1
    """))
    lengths["range"] = lengths.bucket.map(
        lambda b: "4,000 (capped)" if b == 12 else f"{700 + (b - 1) * 300:,}–{700 + b * 300 - 1:,}")
    lengths["capped"] = lengths.bucket == 12
    st.altair_chart(
        alt.Chart(lengths).mark_bar().encode(
            x=alt.X("range:N", sort=list(lengths.range), title="Description length (characters)"),
            y=alt.Y("n:Q", title="Jobs"),
            color=alt.condition("datum.capped", alt.value(ACCENT), alt.value(MUTED)),
            tooltip=["range", "n"],
        ).properties(height=320),
        use_container_width=True)
    st.markdown("The floor at 700 and the spike at 4,000 are both load-time choices, not "
                "properties of the postings. The cut usually falls in the qualifications "
                "section, which is the part that decides fit.")

with tab_audit:
    audit_all = num(query("""
        select entity, check_id, description, n_affected, n_total,
               round(100.0 * n_affected / n_total, 1) as pct, severity, action
        from analytics.audit_checks order by entity, n_affected desc
    """))
    st.dataframe(audit_all, hide_index=True, use_container_width=True)

with tab_dict:
    st.dataframe(num(query("""
        select view_name, column_name, data_type, meaning, null_pct, cleaning
        from analytics.data_dictionary order by view_name, column_name
    """)), hide_index=True, use_container_width=True)

with tab_split:
    st.markdown("20% of each stratum is reserved; rows with identical text share a split. "
                "Assignment is deterministic (salted md5), so rebuilding never reshuffles it.")
    st.dataframe(num(query("""
        select entity, stratum, n_train, n_test,
               round(100.0 * n_test / (n_train + n_test), 1) as test_pct
        from analytics.split_summary order by entity, n_train desc
    """)), hide_index=True, use_container_width=True)
