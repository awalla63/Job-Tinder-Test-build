# Job Tinder

A two-sided "Tinder for jobs" **web** application. Job seekers upload a resume and answer
a short preference survey; employers post a job and answer the mirror survey. Each side
swipes through ranked cards drawn from the other side, and a mutual like creates a match
that exposes an apply link.

```
apps/web/            React 18 + Vite + TypeScript + Tailwind + motion
services/api/        FastAPI — auth, decks, swipes, matches, ML inference
ml/                  encoder, features, training, evaluation
data/                corpus loaders + USAJOBS connector
supabase/migrations/ schema, pgvector, RLS
```

Request path for a deck:
`survey hard filters (SQL)` → `pgvector ANN top-200` → `pairwise features` →
`LightGBM score` → `rank, drop already-swiped` → `cards`

---

## Setup

### 1. Supabase

Create a project, then run `supabase/migrations/0001_init.sql` in the SQL editor. It
enables `pgvector`, creates the schema, and turns on RLS for every table.

Copy `.env.example` to `.env` and fill in `SUPABASE_URL`, `SUPABASE_PROJECT_REF`,
`SUPABASE_SERVICE_ROLE_KEY` and `DATABASE_URL`.

> The service role key bypasses RLS. It belongs in the backend's `.env` only — never in
> `apps/web`, which uses the anon key.

### 2. Python

```bash
python -m pip install -r requirements.txt
```

### 3. Load the corpus

```bash
python data/load_postings.py --csv path/to/postings.csv.zip   # ~44k postings
python data/load_resumes.py  --csv path/to/Resume.csv.zip     # 2,483 demo candidates
python ml/embed.py --target all                               # 5–15 min on CPU
```

`load_postings.py --dry-run` prints the filter chain without writing anything.

Optional live listings (needs a free key from
[developer.usajobs.gov](https://developer.usajobs.gov/apirequest/) — the `User-Agent`
must be the email you registered with):

```bash
python data/refresh_jobs.py --keywords "data analyst,software engineer"
python ml/embed.py --target jobs        # embeds whatever is new
```

### 4. Train

```bash
python ml/train.py              # trains, evaluates, saves to models/
python ml/train.py --ablate     # compares feature families, saves nothing
```

Downloads the labelled dataset on first run. If the model fails its gate (below), delete
`models/fit_model.txt` — the API falls back to cosine similarity and stays fully usable.

### 5. Run

```bash
uvicorn services.api.main:app --reload          # http://localhost:8000
cd apps/web && cp .env.example .env.local && npm install && npm run dev
```

`GET /health` reports whether the fit model loaded and which features it expects.

---

## How matching works

### Data

| Source | What it gives us |
|---|---|
| `postings.csv` (Kaggle, LinkedIn 2024) | 123,849 postings → **44,138** after filtering |
| `Resume.csv` (Kaggle, livecareer) | 2,484 resumes across 24 categories → demo candidates |
| [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) | 8,000 labelled pairs: No Fit / Potential Fit / Good Fit |

### Trimming the corpus

The Supabase free tier caps the database at 500 MB **including indexes**, and the raw
descriptions alone are 467 MB. But most of what gets dropped is genuine fat:

| Step | Remaining |
|---|---|
| All rows | 123,849 |
| Drop missing title, description < 700 chars, Volunteer/Other, closed | 118,268 |
| **Drop duplicate `(company, title)`** — staffing-agency reposts | 92,656 |
| Require a known experience level | 70,326 |
| **Cap 10 postings per company** | **44,138** (141 MB) |

Dedupe removes 21% of the corpus on its own. That matters for ranking more than for
storage: without it, one agency's reposts flood a deck and the app looks broken.

### The model

LightGBM multiclass over the three fit labels, collapsed to a displayable 0–100 score via
the ordinal expectation `0.5·P(Potential) + 1.0·P(Good)` — which keeps the ordering the
annotation encodes instead of discarding the middle class.

**What the labelled dataset actually is.** 8,000 rows, but only **643 unique resumes and
351 unique job descriptions** — a sparse sample of a 643×351 cross-product. The effective
sample is far smaller than the row count suggests, so the model is deliberately
regularised (shallow trees, high `min_child_samples`).

**Its split is a cold-JD split.** Measured: 0% of test JDs appear in train, but **99.9%
of test resumes do**. That matches our primary generalisation axis — every one of our
44k postings is an unseen JD — so the split is usable as-is. But it says nothing about
new resumes, which users upload constantly, so `ml/train.py` reports a `GroupKFold`
cold-resume split alongside it.

**The gate.** Raw TF-IDF cosine gets AUC **0.658** on Good-vs-No Fit. A model that can't
beat that isn't earning its complexity. Current model: **0.683**.

**Why there are no length features.** The first trained model ranked `job_len_log` as its
single most important feature. With only 351 unique JDs, a per-document constant like
length works as a document ID — the model memorises that document's base rate instead of
learning fit, and that memorisation cannot transfer to unseen postings. Dropping the
length proxies raised cold-JD AUC from 0.678 to 0.693. `ml/train.py --ablate` reproduces
the comparison.

**Why survey answers aren't model features.** The labelled data has three columns:
resume text, JD text, label. No salary, no location, no work type. Anything derived from
those would exist at serving time and not at training time — textbook train/serve skew.
Preferences are applied as **hard filters in SQL** instead, and the model answers only
"does this background fit these requirements?"

**Null tolerance.** Only 29% of postings publish a salary. Every filter treats NULL as
*pass*, and features pass NaN to LightGBM rather than imputing zeros — a zero would
assert a salary of $0. Writing `normalized_salary >= x` the obvious way silently discards
71% of the corpus with no error; `test_ranking_filters.py` pins this down.

### Matching rule

Sourced listings (LinkedIn seed, USAJOBS) have no employer account behind them, so
requiring a mutual like would mean nobody ever matches. A seeker's like on a sourced job
creates a match immediately — it's a public posting with a real apply URL. Employer-posted
jobs are the genuine two-sided case and need both sides. One code path, branching on
`jobs.employer_id IS NULL`.

Demo profiles from `Resume.csv` solve employer-side cold start: without them the first
employer to sign up sees an empty deck. They have no auth user, so they can be swiped on
but never match back.

---

## Testing

```bash
python -m pytest        # 54 tests
```

Covers hard-filter null tolerance, both branches of the match rule, feature extraction
and NaN handling, the cosine fallback, resume parsing, the USAJOBS field mapping, and
auth rejection on every data route.

Manual end-to-end check:

1. Sign up as a seeker → upload a resume → complete the survey → confirm a non-empty
   ranked deck.
2. Like a sourced job → a match appears with a working apply link.
3. Second account as employer → post a job → see demo candidates → like one who liked
   that posting → confirm the match fires on both sides.
4. Sanity-check ranking: an `INFORMATION-TECHNOLOGY` resume should not surface `CHEF`
   postings in its top 20. The `Resume.csv` profiles are verified disjoint from the
   training data (0 normalized overlap), so this is a real held-out check.
5. Complete a full swipe session using only the keyboard.
6. `select pg_total_relation_size('public.jobs')` to confirm the real footprint against
   the 500 MB cap.

---

## Notes

- **Embeddings** use `BAAI/bge-small-en-v1.5` (384-dim, MIT) rather than
  all-MiniLM-L6-v2 — same dimensions and CPU cost, much better retrieval. Its
  `max_seq_length` is 256 tokens and it **truncates silently**, so job descriptions
  (~3,775 chars average) are chunked into ~200-token windows and mean-pooled. Naive
  encoding would discard 60–70% of each posting, usually the requirements section.
- **Storage** uses `halfvec(384)`, which halves vector and HNSW index size for ~1–2%
  recall loss. Measure with `pg_total_relation_size` rather than trusting estimates.
- **Resume parsing** uses pdfplumber (MIT) + pypdf (BSD), deliberately not PyMuPDF
  (AGPL-3.0, whose network-use clause reaches a hosted app), and docx2python rather than
  python-docx, which silently skips table cells and text boxes where resume templates
  routinely put the skills sidebar.
- **USAJOBS** terms require crediting the source, not altering values, and sending
  applicants to the official announcement. The job card renders that attribution.
- The Supabase free tier **pauses a project after 7 days of inactivity**; restore it from
  the dashboard.

## Status

Built and verified: schema, loaders, USAJOBS connector, encoder, feature pipeline,
trained model clearing its gate, full API, full web client, 54 passing tests.

Not yet run against a live database — the loaders and migration need a Supabase project
and credentials, which is step 1 above.

---

## Background

> This project intends to create a web application that will treat job searching as a
> Tinder-like experience. Job seekers will be matched to jobs based on their resume and
> initial survey questions. Employers will also be able to swipe on candidates based on
> resume and initial survey questions. There should be matching based on resume and job
> description as well as survey questions. The recommendation system should be based on
> these matches. Candidates and employers should be able to like each other. Once there
> is a mutual like, there should be a link to apply. Finding real users may be difficult,
> especially on the employer side, but [the focus is] on getting the application working.

The employer-side liquidity problem called out there is why the app ships with demo
candidate profiles and the sourced-job auto-match rule described above — both exist so
the product works before it has users on either side.
