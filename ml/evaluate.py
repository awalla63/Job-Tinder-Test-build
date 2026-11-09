"""Model evaluation: a baseline and the candidate fit models, under one protocol.

Every model is scored on the same data, the same two splits and the same metrics, so
the numbers in the comparison table are directly comparable:

    Models
        tfidf_cosine         baseline: raw TF-IDF cosine, no training (the train.py gate)
        embedding_cosine     baseline: raw sentence-embedding cosine, no training
        logistic_regression  candidate: multinomial logistic regression, pairwise_plus
        lightgbm_all         candidate: LightGBM on all 15 features
        lightgbm_pairwise    candidate: LightGBM on pairwise_plus (the shipped model)

    Splits (data: cnamuangtoun/resume-job-description-fit, 8,000 labelled pairs)
        cold_jd       the dataset's own train/test split; no test JD appears in train
        cold_resume   5-fold GroupKFold by resume; no held-out resume appears in train

    Metrics
        auc_good_vs_no       ROC AUC, Good Fit vs No Fit (the train.py gate metric)
        macro_f1             3-class macro F1
        ndcg_seeker_at_10    mean NDCG@10 ranking jobs for one resume (the seeker deck)
        ndcg_employer_at_10  mean NDCG@10 ranking resumes for one JD (the employer deck)
        auc_ci               95% cluster-bootstrap CI (resampling JDs or resumes)

The untrained baselines produce a score, not a class. To give them a macro F1 they are
cut into three classes at the score quantiles that reproduce the training label mix,
which is the most generous threshold rule available without looking at the test labels.

Error analysis breaks the cold-resume out-of-fold predictions (all 8,000 pairs, each
predicted by a model that never saw that resume) down by job seniority, stated years
required, resume experience, skill coverage and embedding similarity.

    python ml/evaluate.py                # full run, writes models/evaluation.json
    python ml/evaluate.py --boot 200     # fewer bootstrap resamples, faster
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler, normalize  # noqa: E402

from ml.features import (  # noqa: E402
    ALL_FEATURE_NAMES,
    FEATURE_SETS,
    LABEL_TO_INT,
    LABELS,
    JobSide,
    ResumeSide,
    build_matrix,
    compute_skill_idf,
    fit_score,
)
from ml.train import _ndcg_by_group, embed_unique, ensure_dataset, make_params  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
GOOD, NO = LABEL_TO_INT["Good Fit"], LABEL_TO_INT["No Fit"]
COL = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# Models. Each fit() returns a predict(X) -> (score, predicted_class) callable.
# ---------------------------------------------------------------------------
def _prior_cuts(train_scores: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Score thresholds that split training rows into the training label proportions."""
    shares = np.bincount(y_train, minlength=3) / len(y_train)
    return np.quantile(train_scores, np.cumsum(shares)[:2])


def cosine_model(column: str):
    def fit(X, y, seed):
        cuts = _prior_cuts(X[:, COL[column]], y)

        def predict(Xp):
            s = Xp[:, COL[column]]
            return s, np.searchsorted(cuts, s, side="right")
        return predict
    return fit


def logistic_model(feature_set: str):
    cols = [COL[c] for c in FEATURE_SETS[feature_set]]

    def fit(X, y, seed):
        # NaN means "not stated" (no years requirement, no seniority word). The indicator
        # columns keep that information instead of pretending the median was stated.
        pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
        pipe.fit(X[:, cols], y)

        def predict(Xp):
            p = pipe.predict_proba(Xp[:, cols])
            return fit_score(p), p.argmax(axis=1)
        return predict
    return fit


def lightgbm_model(feature_set: str, rounds: int):
    names = FEATURE_SETS[feature_set]
    cols = [COL[c] for c in names]

    def fit(X, y, seed):
        booster = lgb.train(make_params(seed),
                            lgb.Dataset(X[:, cols], label=y, feature_name=names),
                            num_boost_round=rounds)

        def predict(Xp):
            p = booster.predict(Xp[:, cols])
            return fit_score(p), p.argmax(axis=1)
        return predict
    return fit


def make_models(rounds: int) -> dict:
    return {
        "tfidf_cosine": ("baseline", cosine_model("tfidf_cosine")),
        "embedding_cosine": ("baseline", cosine_model("emb_cosine")),
        "logistic_regression": ("candidate", logistic_model("pairwise_plus")),
        "lightgbm_all": ("candidate", lightgbm_model("all", rounds)),
        "lightgbm_pairwise": ("candidate", lightgbm_model("pairwise_plus", rounds)),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auc_good_vs_no(y: np.ndarray, s: np.ndarray) -> float:
    mask = np.isin(y, [GOOD, NO])
    if len(set(y[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask] == GOOD, s[mask]))


def cluster_bootstrap_auc(y, s, clusters, reps: int, seed: int) -> list[float]:
    """95% CI for AUC, resampling whole clusters (JDs or resumes), not rows.

    Rows that share a JD or resume are not independent, so a row bootstrap would give
    an interval that is too narrow.
    """
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(clusters)
    members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    values = []
    for _ in range(reps):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in pick])
        a = auc_good_vs_no(y[idx], s[idx])
        if a == a:
            values.append(a)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def paired_auc_gap_ci(y, s_a, s_b, clusters, reps: int, seed: int) -> list[float]:
    """95% cluster-bootstrap CI for AUC(a) - AUC(b) on the same resamples."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(clusters)
    members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    gaps = []
    for _ in range(reps):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in pick])
        a, b = auc_good_vs_no(y[idx], s_a[idx]), auc_good_vs_no(y[idx], s_b[idx])
        if a == a and b == b:
            gaps.append(a - b)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def metrics(y, score, pred, jd_ids, resume_ids) -> dict:
    ndcg_seeker, _ = _ndcg_by_group(resume_ids, y, score)
    ndcg_employer, _ = _ndcg_by_group(jd_ids, y, score)
    recall = confusion_matrix(y, pred, labels=[0, 1, 2]).diagonal() / np.bincount(y, minlength=3)
    return {
        "auc_good_vs_no": round(auc_good_vs_no(y, score), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "ndcg_seeker_at_10": ndcg_seeker,
        "ndcg_employer_at_10": ndcg_employer,
        "recall_per_class": {n: round(float(r), 4) for n, r in zip(LABELS, recall)},
    }


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
SENIORITY_NAMES = {0: "intern", 1: "junior", 2: "mid", 3: "senior", 4: "principal/manager",
                   5: "director/exec", 6: "director/exec"}


def group_columns(X: np.ndarray) -> dict[str, pd.Series]:
    """Group label per row for each condition the errors are broken down by."""
    def cut(values, bins, labels, missing="not stated"):
        out = pd.cut(pd.Series(values), bins=bins, labels=labels).astype(object)
        return out.where(~pd.isna(values), missing)

    emb = X[:, COL["emb_cosine"]]
    return {
        "job_seniority": pd.Series(X[:, COL["job_seniority"]]).map(SENIORITY_NAMES)
                           .fillna("not stated"),
        "job_required_years": cut(X[:, COL["job_required_years"]], [0, 2, 5, 40],
                                  ["1-2 yrs", "3-5 yrs", "6+ yrs"]),
        "resume_years": cut(X[:, COL["resume_years"]], [0, 2, 5, 10, 45],
                            ["0-2 yrs", "3-5 yrs", "6-10 yrs", "11+ yrs"], "unknown"),
        "skill_coverage": cut(X[:, COL["skill_coverage"]], [-0.01, 0.25, 0.5, 1.0],
                              ["<25%", "25-50%", ">50%"], "job lists no known skills"),
        "embedding_similarity": pd.Series(pd.qcut(emb, 3, labels=["low", "medium", "high"])
                                          .astype(object)),
    }


def error_breakdown(X, y, oof: dict[str, tuple]) -> dict:
    """Per-group AUC, macro F1 and severe-error rate for each model's OOF predictions.

    A severe error is the one a user would notice: a No Fit pair predicted Good Fit, or
    a Good Fit pair predicted No Fit.
    """
    out = {}
    for condition, groups in group_columns(X).items():
        rows = []
        for value in sorted(groups.unique(), key=str):
            m = (groups == value).to_numpy()
            row = {"group": value, "n": int(m.sum()),
                   "good_fit_share": round(float((y[m] == GOOD).mean()), 3)}
            for name, (score, pred) in oof.items():
                severe = ((y[m] == NO) & (pred[m] == GOOD)) | ((y[m] == GOOD) & (pred[m] == NO))
                row[name] = {
                    "auc": round(auc_good_vs_no(y[m], score[m]), 4),
                    "macro_f1": round(float(f1_score(y[m], pred[m], average="macro")), 4),
                    "severe_error_rate": round(float(severe.mean()), 4),
                }
            rows.append(row)
        out[condition] = rows
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_features(all_df: pd.DataFrame) -> np.ndarray:
    """The same features train.py builds: TF-IDF fit on all text, cached embeddings."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=60000, sublinear_tf=True)
    vectorizer.fit(pd.concat([all_df.resume_text, all_df.job_description_text]))
    R = normalize(vectorizer.transform(all_df.resume_text))
    J = normalize(vectorizer.transform(all_df.job_description_text))
    tfidf = np.asarray(R.multiply(J).sum(axis=1)).ravel()

    class _LazyEncoder:
        """Only load the encoder if the embedding cache is missing or stale."""
        def __getattr__(self, attr):
            from ml.encoder import get_encoder
            return getattr(get_encoder(), attr)

    encoder = _LazyEncoder()
    resume_vectors = embed_unique(encoder, all_df.resume_text.tolist(), "resumes")
    job_vectors = embed_unique(encoder, all_df.job_description_text.tolist(), "job descriptions")
    skill_idf = compute_skill_idf(
        list(set(all_df.resume_text)) + list(set(all_df.job_description_text)))

    resumes = [ResumeSide(text=t, embedding=resume_vectors[t]) for t in all_df.resume_text]
    jobs = [JobSide(text=t, embedding=job_vectors[t]) for t in all_df.job_description_text]
    return build_matrix(resumes, jobs, tfidf_cosines=tfidf, skill_idf=skill_idf)


def print_table(title: str, results: dict) -> None:
    print(f"\n{title}")
    print(f"{'model':<22}{'AUC':>8}{'95% CI':>18}{'macro F1':>10}"
          f"{'NDCG seek':>11}{'NDCG emp':>10}{'rec Good':>10}")
    print("-" * 89)
    for name, r in results.items():
        ci = f"[{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]"
        print(f"{name:<22}{r['auc_good_vs_no']:>8.4f}{ci:>18}{r['macro_f1']:>10.4f}"
              f"{r['ndcg_seeker_at_10']:>11.4f}{r['ndcg_employer_at_10']:>10.4f}"
              f"{r['recall_per_class']['Good Fit']:>10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    args = parser.parse_args()
    started = time.time()

    train_df, test_df = ensure_dataset()
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    y = all_df.label.map(LABEL_TO_INT).to_numpy()
    is_test = np.r_[np.zeros(len(train_df), bool), np.ones(len(test_df), bool)]
    jd_ids, resume_ids = all_df.job_description_text.to_numpy(), all_df.resume_text.to_numpy()

    print(f"{len(all_df):,} pairs | {len(set(resume_ids)):,} resumes | {len(set(jd_ids)):,} JDs")
    print("Building features ...")
    X = build_features(all_df)
    models = make_models(args.rounds)

    # --- Split 1: cold-JD (the dataset's own split) ---
    print("Evaluating on the cold-JD split ...")
    cold_jd, cold_jd_scores = {}, {}
    for name, (_, fit) in models.items():
        score, pred = fit(X[~is_test], y[~is_test], args.seed)(X[is_test])
        r = metrics(y[is_test], score, pred, jd_ids[is_test], resume_ids[is_test])
        r["auc_ci"] = cluster_bootstrap_auc(y[is_test], score, jd_ids[is_test],
                                            args.boot, args.seed)
        cold_jd[name], cold_jd_scores[name] = r, score

    # --- Split 2: cold-resume (GroupKFold by resume, out-of-fold predictions) ---
    print(f"Evaluating on the cold-resume split ({args.folds}-fold GroupKFold) ...")
    folds = list(GroupKFold(n_splits=args.folds).split(X, y, resume_ids))
    cold_resume, oof = {}, {}
    for name, (_, fit) in models.items():
        score, pred = np.zeros(len(y)), np.zeros(len(y), int)
        for k, (tr, va) in enumerate(folds, 1):
            # seed + fold matches train.py, so lightgbm_pairwise reproduces metrics.json.
            score[va], pred[va] = fit(X[tr], y[tr], args.seed + k)(X[va])
        r = metrics(y, score, pred, jd_ids, resume_ids)
        r["auc_ci"] = cluster_bootstrap_auc(y, score, resume_ids, args.boot, args.seed)
        cold_resume[name], oof[name] = r, (score, pred)

    print_table("COLD-JD SPLIT (unseen job descriptions, n = %d)" % is_test.sum(), cold_jd)
    print_table("COLD-RESUME SPLIT (unseen resumes, out-of-fold, n = %d)" % len(y), cold_resume)

    # --- Selected model vs the gate baseline: is the gain real? ---
    selected, baseline = "lightgbm_pairwise", "tfidf_cosine"
    gaps = {
        "cold_jd": paired_auc_gap_ci(y[is_test], cold_jd_scores[selected],
                                     cold_jd_scores[baseline], jd_ids[is_test],
                                     args.boot, args.seed),
        "cold_resume": paired_auc_gap_ci(y, oof[selected][0], oof[baseline][0],
                                         resume_ids, args.boot, args.seed),
    }
    print(f"\nAUC gain of {selected} over {baseline}, 95% cluster-bootstrap CI:")
    for split, (lo, hi) in gaps.items():
        print(f"  {split:<12} [{lo:+.4f}, {hi:+.4f}]  "
              f"{'excludes 0' if lo > 0 or hi < 0 else 'includes 0'}")

    # --- Error analysis on the cold-resume OOF predictions ---
    shown = ["tfidf_cosine", "logistic_regression", "lightgbm_pairwise"]
    groups = error_breakdown(X, y, {n: oof[n] for n in shown})
    print("\nERROR ANALYSIS (cold-resume out-of-fold; AUC Good vs No / severe-error rate)")
    for condition, rows in groups.items():
        print(f"\n  {condition}")
        print(f"  {'group':<28}{'n':>6}{'%Good':>7}" + "".join(f"{n[:18]:>22}" for n in shown))
        for row in rows:
            cells = "".join(f"{row[n]['auc']:>13.3f} / {row[n]['severe_error_rate']:<6.3f}"
                            for n in shown)
            print(f"  {str(row['group']):<28}{row['n']:>6}{row['good_fit_share']:>7.2f}{cells}")

    cm = confusion_matrix(y, oof[selected][1], labels=[0, 1, 2])
    print(f"\nConfusion matrix, {selected} (cold-resume OOF; rows = true, cols = predicted)")
    print(f"  {'':<15}" + "".join(f"{n:>15}" for n in LABELS))
    for name, row in zip(LABELS, cm):
        print(f"  {name:<15}" + "".join(f"{v:>15,}" for v in row))

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_seconds": round(time.time() - started, 1),
        "n_pairs": int(len(y)), "n_cold_jd_test": int(is_test.sum()),
        "models": {n: {"role": role, "features": (
            FEATURE_SETS["pairwise_plus"] if n in ("logistic_regression", "lightgbm_pairwise")
            else FEATURE_SETS["all"] if n == "lightgbm_all"
            else ["tfidf_cosine" if n == "tfidf_cosine" else "emb_cosine"])}
            for n, (role, _) in models.items()},
        "selected_model": selected,
        "cold_jd": cold_jd,
        "cold_resume": cold_resume,
        "selected_vs_baseline_auc_gap_ci": gaps,
        "confusion_matrix_cold_resume": {"labels": LABELS, "matrix": cm.tolist()},
        "error_analysis_cold_resume": groups,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved models/evaluation.json ({report['duration_seconds']}s)")


if __name__ == "__main__":
    main()
