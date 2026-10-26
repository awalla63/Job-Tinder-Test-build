"""Train the resume/job fit model.

Data: cnamuangtoun/resume-job-description-fit -- 8,000 pairs labelled
No Fit / Potential Fit / Good Fit.

Two things about that dataset drive how this script is written:

1. It is *not* 8,000 independent observations. There are only 643 unique resumes and
   351 unique job descriptions; the rows are a sparse sample of a 643x351 cross-product
   (median 16 rows per JD, 7 per resume). The effective sample size is small, so the
   model is deliberately regularised -- shallow trees, high min_child_samples -- and we
   dedupe before embedding (994 encodes instead of 16,000).

2. The provided train/test split is a **cold-JD split**: 0% of test job descriptions
   appear in train, but 99.9% of test resumes do. That happens to match our primary
   generalisation axis (all 44k postings are unseen JDs), so the split is usable as-is.
   But it says nothing about new resumes, which users upload constantly -- so we also
   report a GroupKFold-by-resume cold-resume split. Reporting only the provided split
   would overstate real-world performance.

The gate: raw TF-IDF cosine achieves AUC 0.658 on Good-vs-No Fit. A model that cannot
beat that is not earning its complexity, and the honest move is to serve cosine alone
(the API already falls back to exactly that when no model file is present).

    python ml/train.py
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
from sklearn.metrics import classification_report, ndcg_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

from ml.encoder import get_encoder  # noqa: E402
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
    select_columns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw"
MODEL_DIR = REPO_ROOT / "models"

HF_BASE = "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/main"

# TF-IDF cosine baseline, measured on the full 8,000 pairs. See module docstring.
BASELINE_AUC = 0.658


def ensure_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ("train", "test"):
        path = DATA_DIR / f"fit_{split}.csv"
        if not path.exists():
            import httpx

            print(f"Downloading {split}.csv ...")
            with httpx.stream("GET", f"{HF_BASE}/{split}.csv", follow_redirects=True,
                              timeout=120.0) as response:
                response.raise_for_status()
                with path.open("wb") as fh:
                    for chunk in response.iter_bytes():
                        fh.write(chunk)
        paths[split] = path
    return pd.read_csv(paths["train"]), pd.read_csv(paths["test"])


def embed_unique(encoder, texts: list[str], kind: str) -> dict[str, np.ndarray]:
    """Embed each distinct text once and return a lookup.

    643 resumes and 351 JDs, not 8,000 of each -- this is the difference between a
    one-minute step and a twenty-minute one. Cached to disk so feature-set ablations
    don't pay the encoding cost again.
    """
    unique = sorted(set(texts))
    cache = DATA_DIR / f"emb_cache_{kind}.npz"
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        if list(stored["keys"]) == unique:
            print(f"  reusing cached embeddings for {len(unique):,} {kind}")
            return dict(zip(unique, stored["vectors"]))

    print(f"  embedding {len(unique):,} unique {kind} ...", flush=True)
    encode = encoder.encode_resumes if kind == "resumes" else encoder.encode_jobs
    vectors = encode(unique)
    np.savez_compressed(cache, keys=np.array(unique, dtype=object), vectors=vectors)
    return dict(zip(unique, vectors))


def _ndcg_by_group(group_ids, y, scores) -> tuple[float | None, int]:
    """Mean NDCG@10 over groups with at least 5 members to rank."""
    values = []
    frame = pd.DataFrame({"g": group_ids, "y": y, "s": scores})
    for _, chunk in frame.groupby("g"):
        if len(chunk) >= 5:
            values.append(ndcg_score(chunk[["y"]].T.values, chunk[["s"]].T.values, k=10))
    return (round(float(np.mean(values)), 4) if values else None), len(values)


def evaluate(model, X, y, job_ids, resume_ids, label: str) -> dict:
    """Metrics for one split.

    NDCG is reported in both directions because the product consumes the score both
    ways, and they are not interchangeable:
      * ndcg_seeker -- ranking many jobs for one resume. This is the seeker deck, the
        primary surface, and the harder direction.
      * ndcg_employer -- ranking many candidates for one posting (the employer deck).
    A model can look strong on one and mediocre on the other, so reporting a single
    NDCG would hide which half of the product is actually working.
    """
    probabilities = model.predict(X)
    scores = fit_score(probabilities)
    predictions = probabilities.argmax(axis=1)

    good_vs_no = np.isin(y, [LABEL_TO_INT["Good Fit"], LABEL_TO_INT["No Fit"]])
    auc = float("nan")
    if good_vs_no.sum() > 1 and len(set(y[good_vs_no])) == 2:
        auc = roc_auc_score((y[good_vs_no] == LABEL_TO_INT["Good Fit"]).astype(int),
                            scores[good_vs_no])

    report = classification_report(y, predictions, labels=[0, 1, 2],
                                   target_names=LABELS, output_dict=True, zero_division=0)

    ndcg_employer, n_employer = _ndcg_by_group(job_ids, y, scores)
    ndcg_seeker, n_seeker = _ndcg_by_group(resume_ids, y, scores)

    return {
        "split": label,
        "n": int(len(y)),
        "auc_good_vs_no": round(float(auc), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "recall_per_class": {name: round(float(report[name]["recall"]), 4) for name in LABELS},
        "ndcg_seeker_at_10": ndcg_seeker,
        "ndcg_employer_at_10": ndcg_employer,
        "n_seeker_groups": n_seeker,
        "n_employer_groups": n_employer,
    }


def make_params(seed: int) -> dict:
    """Regularised for a small effective sample (643 resumes x 351 JDs)."""
    return {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-cold-resume", action="store_true")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="pairwise_plus",
                        help="Which feature family to train on; see ml/features.py.")
    parser.add_argument("--ablate", action="store_true",
                        help="Train every feature set and print a comparison instead of saving.")
    args = parser.parse_args()
    feature_names = FEATURE_SETS[args.feature_set]

    MODEL_DIR.mkdir(exist_ok=True)
    started = time.time()

    train_df, test_df = ensure_dataset()
    for frame in (train_df, test_df):
        frame["y"] = frame["label"].map(LABEL_TO_INT)
    print(f"train {len(train_df):,} rows | test {len(test_df):,} rows")
    print(f"unique resumes {pd.concat([train_df, test_df]).resume_text.nunique():,} | "
          f"unique JDs {pd.concat([train_df, test_df]).job_description_text.nunique():,}")

    all_df = pd.concat([train_df, test_df], ignore_index=True)

    # --- TF-IDF (fitted on everything; it is unsupervised, so this is not leakage) ---
    print("Fitting TF-IDF ...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=60000, sublinear_tf=True)
    vectorizer.fit(pd.concat([all_df.resume_text, all_df.job_description_text]))

    def tfidf_cosines(frame: pd.DataFrame) -> np.ndarray:
        R = normalize(vectorizer.transform(frame.resume_text))
        J = normalize(vectorizer.transform(frame.job_description_text))
        return np.asarray(R.multiply(J).sum(axis=1)).ravel()

    # --- Embeddings ---
    print("Loading encoder ...")
    encoder = get_encoder()
    resume_vectors = embed_unique(encoder, all_df.resume_text.tolist(), "resumes")
    job_vectors = embed_unique(encoder, all_df.job_description_text.tolist(), "job descriptions")

    skill_idf = compute_skill_idf(
        list(set(all_df.resume_text)) + list(set(all_df.job_description_text))
    )

    def featurise(frame: pd.DataFrame) -> np.ndarray:
        resumes = [ResumeSide(text=t, embedding=resume_vectors[t]) for t in frame.resume_text]
        jobs = [JobSide(text=t, embedding=job_vectors[t]) for t in frame.job_description_text]
        return build_matrix(resumes, jobs,
                            tfidf_cosines=tfidf_cosines(frame), skill_idf=skill_idf)

    print("Building features ...")
    X_train_all, y_train = featurise(train_df), train_df.y.values
    X_test_all, y_test = featurise(test_df), test_df.y.values

    # --- Ablation mode: compare feature families, save nothing ---
    if args.ablate:
        print(f"\n{'feature set':<16}{'n':>4}{'cold-JD AUC':>14}{'macro F1':>11}"
              f"{'NDCG seeker':>12}{'NDCG employer':>14}{'recall Good':>13}")
        print("-" * 84)
        for name in ("all", "pairwise", "pairwise_plus"):
            cols = FEATURE_SETS[name]
            m = lgb.train(
                make_params(args.seed),
                lgb.Dataset(select_columns(X_train_all, cols), label=y_train, feature_name=cols),
                num_boost_round=args.rounds,
            )
            r = evaluate(m, select_columns(X_test_all, cols), y_test,
                         test_df.job_description_text.values,
                         test_df.resume_text.values, name)
            print(f"{name:<16}{len(cols):>4}{r['auc_good_vs_no']:>14.4f}"
                  f"{r['macro_f1']:>11.4f}{(r['ndcg_seeker_at_10'] or 0):>12.4f}"
                  f"{(r['ndcg_employer_at_10'] or 0):>14.4f}"
                  f"{r['recall_per_class']['Good Fit']:>13.4f}")
        print("-" * 84)
        print(f"TF-IDF cosine baseline AUC: {BASELINE_AUC}")
        print("\nAblation only; nothing saved. Re-run without --ablate to train and save.")
        return

    X_train = select_columns(X_train_all, feature_names)
    X_test = select_columns(X_test_all, feature_names)

    # --- Cold-JD split (the one the dataset ships with) ---
    print(f"\nTraining on the provided cold-JD split "
          f"(feature set '{args.feature_set}', {len(feature_names)} features, "
          f"{args.rounds} rounds) ...")
    model = lgb.train(
        make_params(args.seed),
        lgb.Dataset(X_train, label=y_train, feature_name=feature_names),
        num_boost_round=args.rounds,
    )
    cold_jd = evaluate(model, X_test, y_test, test_df.job_description_text.values,
                       test_df.resume_text.values, "cold_jd")

    # --- Cold-resume split (what happens when a new user uploads a resume) ---
    cold_resume = None
    if not args.skip_cold_resume:
        print(f"Cross-validating a cold-resume split ({args.folds} folds) ...")
        X_all = select_columns(featurise(all_df), feature_names)
        y_all = all_df.y.values
        groups = all_df.resume_text.values
        oof = np.zeros((len(all_df), 3))
        for fold, (tr, va) in enumerate(GroupKFold(n_splits=args.folds).split(X_all, y_all, groups), 1):
            fold_model = lgb.train(
                make_params(args.seed + fold),
                lgb.Dataset(X_all[tr], label=y_all[tr], feature_name=feature_names),
                num_boost_round=args.rounds,
            )
            oof[va] = fold_model.predict(X_all[va])
            print(f"  fold {fold}/{args.folds}", end="\r")
        print()

        class _Frozen:
            def predict(self, X):  # evaluate() only needs .predict
                return oof
        cold_resume = evaluate(_Frozen(), X_all, y_all, all_df.job_description_text.values,
                               all_df.resume_text.values, "cold_resume")

    # --- Report ---
    importance = sorted(
        zip(feature_names, model.feature_importance("gain")), key=lambda kv: -kv[1]
    )

    print("\n" + "=" * 68)
    print(f"{'metric':<22}{'cold-JD (shipped split)':>24}{'cold-resume (CV)':>22}")
    print("-" * 68)

    def line(name: str, a, b, fmt="{:.4f}"):
        fa = fmt.format(a) if isinstance(a, (int, float)) and a == a else "-"
        fb = fmt.format(b) if isinstance(b, (int, float)) and b == b else "-"
        print(f"{name:<22}{fa:>24}{fb:>22}")

    cr = cold_resume or {}
    line("AUC (Good vs No)", cold_jd["auc_good_vs_no"], cr.get("auc_good_vs_no"))
    line("macro F1", cold_jd["macro_f1"], cr.get("macro_f1"))
    line("NDCG@10 seeker", cold_jd["ndcg_seeker_at_10"], cr.get("ndcg_seeker_at_10"))
    line("NDCG@10 employer", cold_jd["ndcg_employer_at_10"], cr.get("ndcg_employer_at_10"))
    for name in LABELS:
        line(f"  recall {name}", cold_jd["recall_per_class"][name],
             (cr.get("recall_per_class") or {}).get(name))
    print("=" * 68)

    print("\nTop features by gain:")
    for name, gain in importance[:8]:
        print(f"  {name:<24}{gain:>12,.0f}")

    auc = cold_jd["auc_good_vs_no"]
    passed = auc > BASELINE_AUC
    print(f"\nGate: cold-JD AUC {auc:.4f} vs TF-IDF cosine baseline {BASELINE_AUC:.4f} -> "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        print("  The model is not beating raw cosine. Do not ship it: delete "
              "models/fit_model.txt and the API will serve cosine + hard filters, which\n"
              "  is a fully functional product. Revisit ml/features.py before retraining.")
    if cold_resume and cold_resume["auc_good_vs_no"] == cold_resume["auc_good_vs_no"]:
        gap = auc - cold_resume["auc_good_vs_no"]
        if gap > 0.08:
            print(f"  NOTE: cold-JD beats cold-resume by {gap:.3f}. That gap means the model "
                  "leans on\n  resume-specific quirks; lean harder on pair-interaction features.")

    # --- Persist ---
    import joblib

    model.save_model(str(MODEL_DIR / "fit_model.txt"))
    # The feature list is saved alongside the model: the API must build its columns in
    # exactly this order, and a silent mismatch here would mis-rank everything.
    joblib.dump({"vectorizer": vectorizer, "skill_idf": skill_idf,
                 "feature_names": feature_names},
                MODEL_DIR / "fit_artifacts.joblib")

    metrics = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_seconds": round(time.time() - started, 1),
        "model": "lightgbm-multiclass",
        "encoder": encoder.model.__class__.__name__,
        "feature_set": args.feature_set,
        "feature_names": feature_names,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "baseline_auc": BASELINE_AUC,
        "gate_passed": bool(passed),
        "cold_jd": cold_jd,
        "cold_resume": cold_resume,
        "feature_importance": {name: float(gain) for name, gain in importance},
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved models/fit_model.txt, models/fit_artifacts.joblib, models/metrics.json "
          f"({metrics['duration_seconds']}s)")


if __name__ == "__main__":
    main()
