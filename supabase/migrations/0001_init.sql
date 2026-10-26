-- Job Tinder — initial schema
--
-- Storage note: the Supabase free tier caps the database at 500 MB *including indexes*.
-- Embeddings are halfvec(384), not vector(384): halfvec is 16-bit, halving both column and
-- HNSW index storage for ~1-2% recall loss. With ~44k postings that is the difference between
-- comfortable headroom and running up against the cap.

create extension if not exists vector with schema extensions;

-- ---------------------------------------------------------------------------
-- Enums
-- ---------------------------------------------------------------------------
do $$ begin
    create type user_role as enum ('seeker', 'employer');
exception when duplicate_object then null; end $$;

do $$ begin
    create type job_source as enum ('linkedin_seed', 'usajobs', 'employer');
exception when duplicate_object then null; end $$;

do $$ begin
    create type swipe_direction as enum ('like', 'pass');
exception when duplicate_object then null; end $$;

do $$ begin
    create type swipe_target as enum ('job', 'seeker');
exception when duplicate_object then null; end $$;

-- ---------------------------------------------------------------------------
-- profiles
-- ---------------------------------------------------------------------------
create table if not exists public.profiles (
    id           uuid primary key references auth.users (id) on delete cascade,
    role         user_role   not null,
    display_name text        not null default '',
    created_at   timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- seeker_profiles
--
-- is_demo marks the 2,484 Resume.csv rows loaded to solve employer-side cold start.
-- Demo rows have no auth.users row, so user_id is nullable and we key on id instead.
-- ---------------------------------------------------------------------------
create table if not exists public.seeker_profiles (
    id                 uuid primary key default gen_random_uuid(),
    user_id            uuid unique references auth.users (id) on delete cascade,
    is_demo            boolean     not null default false,
    display_name       text        not null default '',
    headline           text,
    resume_text        text        not null default '',
    resume_path        text,
    years_experience   numeric,
    predicted_category text,
    embedding          halfvec(384),
    created_at         timestamptz not null default now(),
    updated_at         timestamptz not null default now(),
    -- A real seeker must have an auth user; a demo profile must not.
    constraint seeker_demo_xor_user check ((is_demo and user_id is null) or (not is_demo and user_id is not null))
);

-- ---------------------------------------------------------------------------
-- employer_profiles
-- ---------------------------------------------------------------------------
create table if not exists public.employer_profiles (
    id           uuid primary key default gen_random_uuid(),
    user_id      uuid unique not null references auth.users (id) on delete cascade,
    company_name text        not null,
    created_at   timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- jobs
--
-- employer_id null  => sourced listing (LinkedIn seed / USAJOBS). Public, has a real
--                      application_url, and a seeker like auto-matches (see match rule).
-- employer_id set   => posted by a registered employer. Requires a mutual like.
-- ---------------------------------------------------------------------------
create table if not exists public.jobs (
    id                         uuid primary key default gen_random_uuid(),
    source                     job_source  not null,
    external_id                text,
    employer_id                uuid references public.employer_profiles (id) on delete cascade,
    company_name               text        not null default '',
    title                      text        not null,
    description                text        not null,
    location                   text,
    state                      text,
    normalized_salary          numeric,
    pay_period                 text,
    formatted_work_type        text,
    formatted_experience_level text,
    remote_allowed             boolean     not null default false,
    application_url            text,
    posted_at                  timestamptz,
    embedding                  halfvec(384),
    created_at                 timestamptz not null default now(),
    unique (source, external_id)
);

-- ---------------------------------------------------------------------------
-- surveys
--
-- answers holds the raw questionnaire; the typed columns below are the denormalised
-- hard filters the deck query actually uses. NULL in any of them means "no preference"
-- and must never narrow the candidate set -- see the null-tolerance note in ranking.py.
-- ---------------------------------------------------------------------------
create table if not exists public.surveys (
    user_id           uuid primary key references auth.users (id) on delete cascade,
    role              user_role   not null,
    answers           jsonb       not null default '{}'::jsonb,
    min_salary        numeric,
    remote_pref       text,                       -- 'remote' | 'onsite' | 'any'
    work_types        text[],
    states            text[],
    categories        text[],
    experience_levels text[],
    updated_at        timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- swipes
-- ---------------------------------------------------------------------------
create table if not exists public.swipes (
    id             uuid primary key default gen_random_uuid(),
    actor_user_id  uuid            not null references auth.users (id) on delete cascade,
    actor_role     user_role       not null,
    target_type    swipe_target    not null,
    target_id      uuid            not null,
    direction      swipe_direction not null,
    score_at_swipe real,
    created_at     timestamptz     not null default now(),
    unique (actor_user_id, target_type, target_id)
);

-- ---------------------------------------------------------------------------
-- matches
-- ---------------------------------------------------------------------------
create table if not exists public.matches (
    id                uuid primary key default gen_random_uuid(),
    seeker_user_id    uuid        not null references auth.users (id) on delete cascade,
    job_id            uuid        not null references public.jobs (id) on delete cascade,
    employer_user_id  uuid references auth.users (id) on delete cascade,
    application_url   text,
    created_at        timestamptz not null default now(),
    unique (seeker_user_id, job_id)
);

-- ---------------------------------------------------------------------------
-- training_runs
-- ---------------------------------------------------------------------------
create table if not exists public.training_runs (
    id            uuid primary key default gen_random_uuid(),
    created_at    timestamptz not null default now(),
    n_examples    integer,
    metrics       jsonb,
    artifact_path text,
    notes         text
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
-- Cosine distance is <=>; <-> is L2 and <#> is negative inner product.
-- Built after bulk load in the loaders (building an HNSW graph on an empty table then
-- inserting 44k rows is far slower than loading first and indexing once).
create index if not exists jobs_embedding_hnsw
    on public.jobs using hnsw (embedding halfvec_cosine_ops)
    with (m = 16, ef_construction = 64);

create index if not exists seeker_embedding_hnsw
    on public.seeker_profiles using hnsw (embedding halfvec_cosine_ops)
    with (m = 16, ef_construction = 64);

-- Hard-filter support
create index if not exists jobs_filters_idx
    on public.jobs (formatted_experience_level, formatted_work_type, remote_allowed, state);
create index if not exists jobs_salary_idx      on public.jobs (normalized_salary);
create index if not exists jobs_employer_idx    on public.jobs (employer_id);
create index if not exists swipes_actor_idx     on public.swipes (actor_user_id, target_type);
create index if not exists matches_seeker_idx   on public.matches (seeker_user_id);
create index if not exists matches_employer_idx on public.matches (employer_user_id);
create index if not exists seeker_demo_idx      on public.seeker_profiles (is_demo);

-- ---------------------------------------------------------------------------
-- Row Level Security
--
-- auth.uid() is wrapped in a sub-select throughout: that makes Postgres evaluate it once
-- per statement as an initPlan rather than once per row, which matters on the swipe and
-- job tables. The FastAPI backend connects with the service role key and bypasses all of
-- this -- these policies protect the path where the browser talks to PostgREST directly.
-- ---------------------------------------------------------------------------
alter table public.profiles          enable row level security;
alter table public.seeker_profiles   enable row level security;
alter table public.employer_profiles enable row level security;
alter table public.jobs              enable row level security;
alter table public.surveys           enable row level security;
alter table public.swipes            enable row level security;
alter table public.matches           enable row level security;
alter table public.training_runs     enable row level security;

-- profiles: own row
create policy "profiles: read own"   on public.profiles for select to authenticated
    using ((select auth.uid()) = id);
create policy "profiles: insert own" on public.profiles for insert to authenticated
    with check ((select auth.uid()) = id);
create policy "profiles: update own" on public.profiles for update to authenticated
    using ((select auth.uid()) = id) with check ((select auth.uid()) = id);

-- seeker_profiles: a seeker manages their own; employers read via the backend only.
create policy "seeker: read own"   on public.seeker_profiles for select to authenticated
    using ((select auth.uid()) = user_id);
create policy "seeker: insert own" on public.seeker_profiles for insert to authenticated
    with check ((select auth.uid()) = user_id);
create policy "seeker: update own" on public.seeker_profiles for update to authenticated
    using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);

-- employer_profiles: own row
create policy "employer: read own"   on public.employer_profiles for select to authenticated
    using ((select auth.uid()) = user_id);
create policy "employer: insert own" on public.employer_profiles for insert to authenticated
    with check ((select auth.uid()) = user_id);
create policy "employer: update own" on public.employer_profiles for update to authenticated
    using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);

-- jobs: every authenticated user may read; writes go through the backend.
-- With RLS on and no write policy, insert/update/delete are denied by default.
create policy "jobs: read all" on public.jobs for select to authenticated using (true);
create policy "jobs: employer manages own" on public.jobs for all to authenticated
    using (employer_id in (select id from public.employer_profiles where user_id = (select auth.uid())))
    with check (employer_id in (select id from public.employer_profiles where user_id = (select auth.uid())));

-- surveys / swipes: own rows
create policy "surveys: own" on public.surveys for all to authenticated
    using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
create policy "swipes: own"  on public.swipes for all to authenticated
    using ((select auth.uid()) = actor_user_id) with check ((select auth.uid()) = actor_user_id);

-- matches: readable by either participant; created by the backend only.
create policy "matches: read own" on public.matches for select to authenticated
    using ((select auth.uid()) in (seeker_user_id, employer_user_id));

-- training_runs: no policy => authenticated clients cannot touch it. Backend only.
