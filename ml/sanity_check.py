"""End-to-end ranking sanity check, without a database.

Runs the real serving stack -- the same encoder, the same ml/features.py, the same
trained model -- over resumes and postings read straight from the CSVs, and asks the one
question that matters before anyone looks at the UI: does an IT resume rank IT jobs above
unrelated ones?

The `Resume.csv` profiles used here were verified to have zero normalised overlap with
the 643 resumes in the training data, so this is a genuine held-out check rather than a
memorisation test.

    python ml/sanity_check.py --postings path/to/postings.csv.zip --resumes path/to/Resume.csv.zip
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.load_postings import open_csv  # noqa: E402
from ml.encoder import get_encoder, job_text  # noqa: E402
from ml.features import JobSide, ResumeSide  # noqa: E402
from services.api.scoring import Scorer  # noqa: E402

csv.field_size_limit(10**9)

# Resume category -> the title keywords a correct ranking should surface for it.
EXPECTATIONS = {
    "INFORMATION-TECHNOLOGY": ["engineer", "developer", "software", "it ", "data",
                               "systems", "technical", "security", "network", "cloud"],
    "ACCOUNTANT": ["account", "financial", "audit", "tax", "bookkeep", "controller",
                   "finance", "payroll"],
    "CHEF": ["chef", "cook", "culinary", "kitchen", "restaurant", "food", "sous"],
    "TEACHER": ["teacher", "instructor", "education", "tutor", "faculty", "professor",
                "school", "curriculum"],
    "HEALTHCARE": ["nurse", "clinical", "patient", "medical", "health", "care",
                   "physician", "therapist"],
}


def load_postings(path: Path, limit: int) -> list[dict]:
    rows = []
    with open_csv(path) as fh:
        for row in csv.DictReader(fh):
            description = row.get("description") or ""
            title = (row.get("title") or "").strip()
            if not title or len(description) < 700:
                continue
            rows.append({
                "title": title,
                "company": (row.get("company_name") or "").strip(),
                "description": description[:4000],
            })
            if len(rows) >= limit:
                break
    return rows


def load_resumes(path: Path, per_category: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    with open_csv(path) as fh:
        for row in csv.DictReader(fh):
            category = (row.get("Category") or "").strip()
            text = (row.get("Resume_str") or "").strip()
            if category in EXPECTATIONS and len(text) > 400 and len(buckets[category]) < per_category:
                buckets[category].append({"category": category, "text": text[:8000]})
    return [r for rows in buckets.values() for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postings", required=True)
    parser.add_argument("--resumes", required=True)
    parser.add_argument("--n-jobs", type=int, default=600)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading {args.n_jobs} postings and resumes ...")
    jobs = load_postings(Path(args.postings), args.n_jobs)
    resumes = load_resumes(Path(args.resumes), args.per_category)
    print(f"  {len(jobs)} postings, {len(resumes)} resumes "
          f"across {len({r['category'] for r in resumes})} categories")

    encoder = get_encoder()
    scorer = Scorer()
    scorer.load()
    print(f"  model loaded: {scorer.ready}"
          f"{'' if scorer.ready else '  (cosine fallback)'}")

    print("Embedding postings ...", flush=True)
    job_texts = [job_text(j["title"], j["company"], j["description"]) for j in jobs]
    job_vectors = encoder.encode_jobs(job_texts)

    print("Embedding resumes ...", flush=True)
    resume_vectors = encoder.encode_resumes([r["text"] for r in resumes])

    print(f"\nScoring each resume against all {len(jobs)} postings "
          f"(top {args.top_k} counted)\n")
    print(f"{'resume category':<26}{'on-target in top-k':>20}{'baseline':>11}{'lift':>8}")
    print("-" * 65)

    per_category_hits: dict[str, list[float]] = defaultdict(list)

    for resume, resume_vector in zip(resumes, resume_vectors):
        keywords = EXPECTATIONS[resume["category"]]
        scores = scorer.score(
            [ResumeSide(text=resume["text"], embedding=resume_vector)] * len(jobs),
            [JobSide(text=t, embedding=v) for t, v in zip(job_texts, job_vectors)],
        )
        order = np.argsort(-scores)[: args.top_k]
        hits = sum(
            1 for i in order if any(k in jobs[i]["title"].lower() for k in keywords)
        )
        per_category_hits[resume["category"]].append(hits / args.top_k)

    # How often a random posting would be on-target, as the comparison point.
    baselines = {
        category: sum(
            1 for j in jobs if any(k in j["title"].lower() for k in keywords)
        ) / len(jobs)
        for category, keywords in EXPECTATIONS.items()
    }

    overall_rate, overall_base = [], []
    for category, rates in sorted(per_category_hits.items()):
        rate = float(np.mean(rates))
        base = baselines[category]
        lift = rate / base if base else float("inf")
        overall_rate.append(rate)
        overall_base.append(base)
        flag = "" if lift >= 1.5 else "   <-- weak"
        print(f"{category:<26}{rate:>19.0%}{base:>11.0%}{lift:>7.1f}x{flag}")

    print("-" * 65)
    mean_rate, mean_base = float(np.mean(overall_rate)), float(np.mean(overall_base))
    print(f"{'MEAN':<26}{mean_rate:>19.0%}{mean_base:>11.0%}"
          f"{mean_rate / mean_base:>7.1f}x")

    print(
        "\nLift is how much more often an on-target posting appears in the top-k than\n"
        "random selection would produce. Above ~2x means the ranking is doing real work;\n"
        "around 1x means it is no better than chance and something upstream is wrong."
    )


if __name__ == "__main__":
    main()
