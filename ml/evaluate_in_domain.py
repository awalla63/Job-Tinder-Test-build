"""In-domain check: do the models rank the app's own jobs sensibly for the app's own resumes?

ml/evaluate.py scores models on the labelled fit dataset, which is not the data the app
serves. No fit labels exist for the app's corpus, so this script uses two proxy labels
that do exist there:

    category match       a posting is on-target for a resume when its title matches the
                         resume's Kaggle category (CATEGORY_TITLES below); e.g. a CHEF
                         resume and a "Sous Chef" posting
    seniority mismatch   a severe mismatch is a resume with <= 2 years of experience shown
                         a Director/Executive posting, or one with >= 10 years shown an
                         Internship/Entry level posting (LinkedIn's own level field)

Data: the reserved test split from analytics/prepare.py (`split = 'test'`) -- 507 demo
resumes and 8,860 postings, with the embeddings the app actually stores. None of it was
used to train any model (they train on cnamuangtoun/resume-job-description-fit) or in the
EDA charts. Pulled once from Supabase and cached in data/raw/.

Protocol, mirroring the live seeker deck (services/api/ranking.py): for each resume,
retrieve the 200 nearest postings by embedding cosine, then let each model reorder those
200. Every model reranks the same candidates, and the candidate models are trained exactly
as in ml/evaluate.py (cold-JD train rows, same seeds), so the two scripts compare the same
five models.

Job pool (--jobs): the LinkedIn test postings above, or live USAJOBS postings fetched
from the API and embedded locally (nothing is written to the database):

    usajobs        JobSummary only: the short text data/usajobs_client.py used to store
    usajobs_full   the connector's description as stored now: JobSummary + MajorDuties +
                   QualificationSummary, capped at 4,000 characters like LinkedIn's

USAJOBS has no LinkedIn experience level, so the seniority check reads the GS grade
instead (GS_LEVELS below): GS-7 and under counts as entry level, GS-14 and up or SES as
director level. Postings on other pay plans (wage grade, VA, etc.) are left unlevelled.

    python ml/evaluate_in_domain.py                      # needs DATABASE_URL the first time
    python ml/evaluate_in_domain.py --refresh            # re-pull the test split from Supabase
    python ml/evaluate_in_domain.py --jobs usajobs       # needs USAJOBS_API_KEY the first time
    python ml/evaluate_in_domain.py --jobs usajobs_full
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
from sklearn.metrics import ndcg_score  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

from ml import features  # noqa: E402
from ml.encoder import job_text  # noqa: E402
from ml.evaluate import build_features, make_models  # noqa: E402
from ml.features import ALL_FEATURE_NAMES, LABEL_TO_INT, JobSide, ResumeSide  # noqa: E402
from ml.train import ensure_dataset  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "data" / "raw" / "in_domain_test.pkl"
USAJOBS_RAW_CACHE = REPO_ROOT / "data" / "raw" / "usajobs_eval_raw.pkl"
USAJOBS_POSTINGS = 3000
GS_PLANS = {"GS", "GL", "GG"}  # General Schedule and its law-enforcement / excepted twins
MODEL_DIR = REPO_ROOT / "models"
COL = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}

RETRIEVE = 200
TOP_K = 10
SELECTED, BASELINE = "lightgbm_pairwise", "tfidf_cosine"

# Resume category -> posting-title pattern that counts as on-target. Matched on the
# lowercased title with word boundaries. A proxy: it rewards being in the right field,
# not fit within the field, and some categories (ARTS, APPAREL) map loosely to titles.
CATEGORY_TITLES = {
    "ACCOUNTANT": r"account(ant|ing)|bookkeep|audit|\btax\b|payroll|controller|accounts (payable|receivable)",
    "ADVOCATE": r"attorney|lawyer|paralegal|legal|counsel|advocate|litigation",
    "AGRICULTURE": r"agricultur|farm|agronom|\bcrop|ranch|grower|horticultur|\bplant manager",
    "APPAREL": r"apparel|fashion|clothing|merchandis|textile|garment|footwear",
    "ARTS": r"\bart\b|\barts\b|artist|museum|music|theat(er|re)|gallery|creative|illustrat",
    "AUTOMOBILE": r"automotive|\bauto\b|vehicle|mechanic|dealership|\bcar\b|collision|fleet",
    "AVIATION": r"aviation|aircraft|pilot|airline|flight|avionic|airport|\ba&p\b|aerospace",
    "BANKING": r"\bbank|teller|\bloan|mortgage|credit|branch manager|lending|underwrit",
    "BPO": r"call center|contact center|customer (service|support|care)|\bbpo\b|representative",
    "BUSINESS-DEVELOPMENT": r"business development|partnership|account executive|growth|\bbdr\b|\bsdr\b",
    "CHEF": r"\bchef|\bcook\b|culinary|kitchen|\bsous\b|food service|restaurant",
    "CONSTRUCTION": r"construction|superintendent|carpenter|electrician|foreman|estimator|plumb|\bhvac\b|site manager",
    "CONSULTANT": r"consult|advisory",
    "DESIGNER": r"design|\bux\b|\bui\b|graphic|creative",
    "DIGITAL-MEDIA": r"digital|media|content|marketing|video|\bseo\b|social",
    "ENGINEERING": r"engineer|mechanical|electrical|civil|manufacturing|process",
    "FINANCE": r"financ|\bfp&a\b|investment|treasury|controller|analyst",
    "FITNESS": r"fitness|personal trainer|\bcoach|\bgym\b|wellness|yoga|exercise",
    "HEALTHCARE": r"nurse|\brn\b|medical|clinical|health|patient|physician|therap|pharmac|\bcare\b",
    "HR": r"\bhr\b|human resources|recruit|talent|people (operations|partner)|benefits",
    "INFORMATION-TECHNOLOGY": r"\bit\b|software|developer|programmer|\bdata\b|network|systems?\b|cloud|devops|cyber|database|help ?desk|technical support|\bweb\b",
    "PUBLIC-RELATIONS": r"public relations|\bpr\b|communications|media relations|publicist",
    "SALES": r"sales|account executive|account manager|business development",
    "TEACHER": r"teacher|instructor|tutor|education|faculty|professor|school|teaching",
}

JUNIOR_LEVELS = {"Internship", "Entry level"}
SENIOR_LEVELS = {"Director", "Executive"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _vector(text: str) -> np.ndarray:
    return np.asarray(json.loads(text), dtype=np.float32)


def pull_test_split() -> dict:
    """The reserved test rows with their stored embeddings, read-only."""
    from data.db import connect

    with connect() as conn:
        conn.execute("set session characteristics as transaction read only")
        resumes = pd.DataFrame(conn.execute("""
            select p.id::text, q.category, p.resume_text, p.embedding::text
            from analytics.resumes_clean q join public.seeker_profiles p using (id)
            where q.split = 'test' and p.embedding is not null order by p.id
        """).fetchall(), columns=["id", "category", "text", "embedding"])
        jobs = pd.DataFrame(conn.execute("""
            select j.id::text, j.title, j.company_name, j.description,
                   j.formatted_experience_level, j.embedding::text
            from analytics.jobs_clean q join public.jobs j using (id)
            where q.split = 'test' and j.embedding is not null order by j.id
        """).fetchall(), columns=["id", "title", "company", "description", "level",
                                  "embedding"])
    return {
        "resumes": resumes.drop(columns="embedding"),
        "resume_vectors": np.stack([_vector(v) for v in resumes.embedding]),
        "jobs": jobs.drop(columns="embedding"),
        "job_vectors": np.stack([_vector(v) for v in jobs.embedding]),
    }


def load_test_split(refresh: bool) -> dict:
    if CACHE.exists() and not refresh:
        return joblib.load(CACHE)
    print("Pulling the reserved test split from Supabase ...")
    data = pull_test_split()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, CACHE)
    return data


def gs_level(descriptor: dict) -> str:
    """Map a posting's federal grade onto the LinkedIn level names the metrics use."""
    plans = {g.get("Code") for g in descriptor.get("JobGrade") or []}
    details = (descriptor.get("UserArea") or {}).get("Details") or {}
    if "ES" in plans:
        return "Director"
    if not plans & GS_PLANS:
        return ""
    try:
        low, high = int(details.get("LowGrade")), int(details.get("HighGrade"))
    except (TypeError, ValueError):
        return ""
    if high <= 7:
        return "Entry level"
    if low >= 14:
        return "Director"
    return "Mid-Senior level"


def load_usajobs(variant: str, refresh: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """Recent USAJOBS postings and their embeddings, cached; never touches the database."""
    from dotenv import load_dotenv

    from data import usajobs_client

    load_dotenv(REPO_ROOT / ".env")
    if refresh or not USAJOBS_RAW_CACHE.exists():
        print(f"Fetching {USAJOBS_POSTINGS:,} postings from the USAJOBS API ...")
        # No keyword: every posting from the last 30 days, so the pool is not tilted
        # toward the fields a keyword list happens to name.
        items = list(usajobs_client.search("", days_posted=30, results_per_page=500,
                                           max_results=USAJOBS_POSTINGS))
        joblib.dump(items, USAJOBS_RAW_CACHE)
    items = joblib.load(USAJOBS_RAW_CACHE)

    rows = []
    for item in items:
        record = usajobs_client.map_record(item)
        if not record:
            continue
        descriptor = item.get("MatchedObjectDescriptor") or {}
        if variant == "usajobs":
            record["description"] = descriptor["UserArea"]["Details"]["JobSummary"]
        rows.append({"id": record["external_id"], "title": record["title"],
                     "company": record["company_name"], "description": record["description"],
                     "level": gs_level(descriptor)})
    jobs = pd.DataFrame(rows).drop_duplicates("id").reset_index(drop=True)

    cache = REPO_ROOT / "data" / "raw" / f"usajobs_eval_{variant}_vectors.npz"
    keys = jobs.id.to_numpy(dtype=object)
    if cache.exists() and not refresh:
        stored = np.load(cache, allow_pickle=True)
        if list(stored["keys"]) == list(keys):
            return jobs, stored["vectors"]
    from ml.encoder import get_encoder

    print(f"Embedding {len(jobs):,} USAJOBS postings ({variant}) ...", flush=True)
    vectors = get_encoder().encode_jobs(
        [job_text(t, c, d) for t, c, d in zip(jobs.title, jobs.company, jobs.description)])
    np.savez_compressed(cache, keys=keys, vectors=vectors)
    return jobs, vectors


def memoise_feature_extractors() -> None:
    """Cache the per-document extractors build_row calls.

    Each resume meets 200 postings and each posting meets many resumes, so without this
    the same skill/seniority/token extraction runs hundreds of times per document. The
    feature code itself is unchanged; only repeated calls on identical text are skipped.
    """
    features.skills.extract = lru_cache(maxsize=None)(features.skills.extract)
    for name in ("required_years", "resume_years", "seniority", "_tokens"):
        setattr(features, name, lru_cache(maxsize=None)(getattr(features, name)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def per_resume_metrics(order: np.ndarray, on_target: np.ndarray, levels: np.ndarray,
                       years: float) -> dict:
    """Metrics for one resume, given its candidates' order under one model."""
    top = order[:TOP_K]
    mismatch = np.nan
    if years == years:
        if years <= 2:
            mismatch = float(np.isin(levels[top], list(SENIOR_LEVELS)).mean())
        elif years >= 10:
            mismatch = float(np.isin(levels[top], list(JUNIOR_LEVELS)).mean())
    relevance = on_target.astype(float)
    ranked_scores = np.zeros(len(order))
    ranked_scores[order] = np.arange(len(order), 0, -1)
    return {
        "p_at_10": float(relevance[top].mean()),
        "ndcg_at_10": float(ndcg_score([relevance], [ranked_scores], k=TOP_K))
        if relevance.any() else np.nan,
        "seniority_mismatch_at_10": mismatch,
    }


def bootstrap_gap(a: np.ndarray, b: np.ndarray, reps: int, seed: int) -> list[float]:
    """95% CI for mean(a - b), resampling resumes (the paired unit)."""
    rng = np.random.default_rng(seed)
    diff = a - b
    diff = diff[~np.isnan(diff)]
    means = [rng.choice(diff, len(diff)).mean() for _ in range(reps)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true",
                        help="re-pull the test split (and USAJOBS postings) instead of the cache")
    parser.add_argument("--jobs", choices=["linkedin", "usajobs", "usajobs_full"],
                        default="linkedin", help="job pool to rank (see module docstring)")
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boot", type=int, default=1000)
    args = parser.parse_args()
    started = time.time()

    data = load_test_split(args.refresh)
    resumes = data["resumes"]
    if args.jobs == "linkedin":
        jobs, job_vectors = data["jobs"], data["job_vectors"]
    else:
        jobs, job_vectors = load_usajobs(args.jobs, args.refresh)
    print(f"{len(resumes):,} test resumes | {len(jobs):,} {args.jobs} postings")

    titles = jobs.title.fillna("").str.lower()
    # Non-capturing groups: str.contains only needs a match, not the groups.
    patterns = {c: re.compile(p.replace("(", "(?:")) for c, p in CATEGORY_TITLES.items()}
    on_target_all = {c: titles.str.contains(p).to_numpy() for c, p in patterns.items()}
    levels = jobs.level.fillna("").to_numpy()
    job_texts = [job_text(t, c, d) for t, c, d in zip(jobs.title, jobs.company,
                                                       jobs.description)]

    # --- Retrieval: the same 200 candidates per resume for every model ---
    sims = data["resume_vectors"] @ job_vectors.T
    candidates = np.argsort(-sims, axis=1)[:, :RETRIEVE]

    # --- Features for every (resume, candidate) pair, exactly as the API builds them ---
    print(f"Building features for {candidates.size:,} pairs ...")
    memoise_feature_extractors()
    artifacts = joblib.load(MODEL_DIR / "fit_artifacts.joblib")
    vectorizer, skill_idf = artifacts["vectorizer"], artifacts["skill_idf"]
    R = normalize(vectorizer.transform(resumes.text))
    J = normalize(vectorizer.transform(job_texts))
    tfidf = (R @ J.T).toarray()

    rows_r = np.repeat(np.arange(len(resumes)), RETRIEVE)
    rows_j = candidates.ravel()
    X = features.build_matrix(
        [ResumeSide(text=resumes.text.iat[i], embedding=data["resume_vectors"][i])
         for i in rows_r],
        [JobSide(text=job_texts[j], embedding=job_vectors[j]) for j in rows_j],
        tfidf_cosines=tfidf[rows_r, rows_j], skill_idf=skill_idf,
    )

    # --- Train the candidates on the labelled data, as ml/evaluate.py does ---
    print("Training the candidate models on the labelled fit data ...")
    train_df, test_df = ensure_dataset()
    fit_df = pd.concat([train_df, test_df], ignore_index=True)
    X_fit = build_features(fit_df)
    y_fit = fit_df.label.map(LABEL_TO_INT).to_numpy()
    train_rows = np.arange(len(train_df))

    scores = {}
    for name, (_, fit) in make_models(args.rounds).items():
        predict = fit(X_fit[train_rows], y_fit[train_rows], args.seed)
        scores[name] = predict(X)[0].reshape(len(resumes), RETRIEVE)

    # --- Score each resume's reranked deck ---
    resume_years = X[::RETRIEVE, COL["resume_years"]]
    per_model = {}
    for name, s in scores.items():
        rows = []
        for i, category in enumerate(resumes.category):
            cand = candidates[i]
            order = np.argsort(-s[i], kind="stable")
            m = per_resume_metrics(order, on_target_all[category][cand], levels[cand],
                                   resume_years[i])
            rows.append({"category": category, **m})
        per_model[name] = pd.DataFrame(rows)

    base_rate = {c: float(v.mean()) for c, v in on_target_all.items()}
    pool_rate = pd.Series([on_target_all[c][candidates[i]].mean()
                           for i, c in enumerate(resumes.category)])

    def summary(frame: pd.DataFrame) -> dict:
        # Macro over categories, so the 24 categories count equally whatever their size.
        by_cat = frame.groupby("category")[["p_at_10", "ndcg_at_10"]].mean()
        return {
            "p_at_10_macro": round(float(by_cat.p_at_10.mean()), 4),
            "ndcg_at_10_macro": round(float(by_cat.ndcg_at_10.mean()), 4),
            "seniority_mismatch_at_10": round(float(frame.seniority_mismatch_at_10.mean()), 4),
        }

    overall = {name: summary(f) for name, f in per_model.items()}
    gaps = {
        f"{SELECTED}_minus_{other}": {
            metric: bootstrap_gap(per_model[SELECTED][metric].to_numpy(),
                                  per_model[other][metric].to_numpy(), args.boot, args.seed)
            for metric in ("p_at_10", "seniority_mismatch_at_10")
        }
        for other in (BASELINE, "embedding_cosine")
    }

    random_macro = float(np.mean([base_rate[c] for c in resumes.category.unique()]))
    pool_macro = float(pool_rate.groupby(resumes.category).mean().mean())
    n_young = int((resume_years <= 2).sum())
    n_senior = int((resume_years >= 10).sum())

    print(f"\nON-TARGET SHARE (category match), macro over {resumes.category.nunique()} "
          f"categories")
    print(f"  random posting                {random_macro:.3f}")
    print(f"  200 retrieved by embedding    {pool_macro:.3f}")
    print(f"\n{'reranker':<22}{'P@10':>8}{'NDCG@10':>10}{'seniority mismatch@10':>24}")
    print("-" * 64)
    for name, r in overall.items():
        print(f"{name:<22}{r['p_at_10_macro']:>8.3f}{r['ndcg_at_10_macro']:>10.3f}"
              f"{r['seniority_mismatch_at_10']:>24.3f}")
    print(f"(seniority mismatch over {n_young} resumes with <= 2 yrs and "
          f"{n_senior} with >= 10 yrs)")
    print("\n95% bootstrap CI over resumes, selected minus other:")
    for pair, metrics_ci in gaps.items():
        for metric, (lo, hi) in metrics_ci.items():
            print(f"  {pair:<46}{metric:<26}[{lo:+.3f}, {hi:+.3f}]")

    cats = pd.DataFrame({
        name: f.groupby("category").p_at_10.mean() for name, f in per_model.items()
    })
    cats.insert(0, "n_resumes", resumes.category.value_counts())
    cats.insert(1, "random", pd.Series(base_rate))
    cats["selected_lift"] = cats[SELECTED] / cats["random"]
    cats = cats.sort_values(SELECTED)
    print("\nP@10 BY RESUME CATEGORY (sorted by the selected model)")
    print(cats.round(3).to_string())

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_seconds": round(time.time() - started, 1),
        "job_source": args.jobs,
        "n_resumes": int(len(resumes)), "n_postings": int(len(jobs)),
        "retrieved_per_resume": RETRIEVE, "top_k": TOP_K,
        "on_target_random_macro": round(random_macro, 4),
        "on_target_retrieved_macro": round(pool_macro, 4),
        "n_resumes_seniority_checked": {"le_2_years": n_young, "ge_10_years": n_senior},
        "overall": overall,
        "selected_gap_ci": gaps,
        "p_at_10_by_category": cats.round(4).reset_index(names="category")
                                   .to_dict(orient="records"),
        "category_title_patterns": CATEGORY_TITLES,
    }
    name = "evaluation_in_domain" + ("" if args.jobs == "linkedin" else f"_{args.jobs}")
    (MODEL_DIR / f"{name}.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved models/{name}.json ({report['duration_seconds']}s)")


if __name__ == "__main__":
    main()
