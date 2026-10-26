import { useCallback, useEffect, useState } from "react";
import { api, type CandidateCard } from "../lib/api";
import { CandidateCardBody } from "../components/Cards";
import { SwipeDeck, type Decision } from "../components/SwipeDeck";

type Job = {
  id: string;
  title: string;
  company_name: string;
  location: string | null;
  likes: number;
  created_at: string;
};

export default function EmployerJobs() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const rows = await api.employerJobs();
      setJobs(rows);
      if (!selected && rows.length) setSelected(rows[0].id);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [selected]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-900">Your postings</h1>
        <button
          onClick={() => setCreating((v) => !v)}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700"
        >
          {creating ? "Cancel" : "Post a job"}
        </button>
      </div>

      {error && <p role="alert" className="mb-4 text-rose-700">{error}</p>}

      {creating && (
        <NewJobForm
          onCreated={() => {
            setCreating(false);
            void load();
          }}
        />
      )}

      {jobs.length === 0 && !creating && (
        <p className="rounded-xl border border-dashed border-slate-300 p-8 text-center text-slate-500">
          No postings yet. Create one to start seeing candidates.
        </p>
      )}

      {jobs.length > 0 && (
        <>
          <div className="mb-8 flex flex-wrap gap-2">
            {jobs.map((job) => (
              <button
                key={job.id}
                onClick={() => setSelected(job.id)}
                aria-pressed={selected === job.id}
                className={`rounded-lg border px-4 py-2 text-left text-sm transition ${
                  selected === job.id
                    ? "border-indigo-500 bg-indigo-50 text-indigo-700"
                    : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
                }`}
              >
                <span className="font-medium">{job.title}</span>
                <span className="ml-2 text-xs text-slate-400">{job.likes} likes</span>
              </button>
            ))}
          </div>

          {selected && <CandidateDeck jobId={selected} />}
        </>
      )}
    </div>
  );
}

function CandidateDeck({ jobId }: { jobId: string }) {
  const [cards, setCards] = useState<CandidateCard[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [matched, setMatched] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setCards(await api.candidateDeck(jobId, 20));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    void load();
  }, [load]);

  const onDecide = useCallback(async (candidate: CandidateCard, decision: Decision) => {
    try {
      const result = await api.swipe({
        target_type: "seeker",
        target_id: candidate.id,
        direction: decision,
        score_at_swipe: candidate.score,
      });
      if (result.matched) setMatched(candidate.display_name);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  if (loading) return <p className="text-slate-600">Ranking candidates…</p>;
  if (error) return <p role="alert" className="text-rose-700">{error}</p>;

  return (
    <>
      {matched && (
        <p role="status" className="mb-4 rounded-lg bg-emerald-50 px-4 py-3 text-sm text-emerald-800">
          Matched with {matched} — they liked this posting too. See the Matches tab.
        </p>
      )}
      <SwipeDeck
        items={cards}
        getKey={(c) => c.id}
        describe={(c) =>
          `${c.display_name}${c.headline ? `, ${c.headline}` : ""}, ${Math.round(c.score)} percent fit`
        }
        renderCard={(c) => <CandidateCardBody candidate={c} />}
        onDecide={onDecide}
        emptyState={
          <div>
            <p className="font-medium text-slate-700">No more candidates for this posting.</p>
            <button onClick={load} className="mt-4 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white">
              Reload
            </button>
          </div>
        }
      />
    </>
  );
}

function NewJobForm({ onCreated }: { onCreated: () => void }) {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [location, setLocation] = useState("");
  const [salary, setSalary] = useState("");
  const [remote, setRemote] = useState(false);
  const [applyUrl, setApplyUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.createJob({
        title,
        description,
        location: location || null,
        state: location.split(",").pop()?.trim().toUpperCase() || null,
        normalized_salary: salary ? Number(salary) : null,
        pay_period: salary ? "YEARLY" : null,
        formatted_work_type: "Full-time",
        remote_allowed: remote,
        application_url: applyUrl || null,
      });
      onCreated();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="mb-8 space-y-4 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div>
        <label htmlFor="job-title" className="block text-sm font-medium text-slate-700">Title</label>
        <input id="job-title" required value={title} onChange={(e) => setTitle(e.target.value)}
               className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" />
      </div>
      <div>
        <label htmlFor="job-desc" className="block text-sm font-medium text-slate-700">
          Description
        </label>
        <textarea
          id="job-desc" required minLength={50} rows={8}
          value={description} onChange={(e) => setDescription(e.target.value)}
          className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2"
        />
        <p className="mt-1 text-xs text-slate-500">
          Include required skills and years of experience — that's what the matcher reads.
        </p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <label htmlFor="job-loc" className="block text-sm font-medium text-slate-700">Location</label>
          <input id="job-loc" placeholder="Charlotte, NC" value={location}
                 onChange={(e) => setLocation(e.target.value)}
                 className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" />
        </div>
        <div>
          <label htmlFor="job-sal" className="block text-sm font-medium text-slate-700">
            Salary (USD/year)
          </label>
          <input id="job-sal" type="number" value={salary} onChange={(e) => setSalary(e.target.value)}
                 className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" />
        </div>
      </div>
      <div>
        <label htmlFor="job-url" className="block text-sm font-medium text-slate-700">
          Application URL
        </label>
        <input id="job-url" type="url" value={applyUrl} onChange={(e) => setApplyUrl(e.target.value)}
               className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" />
      </div>
      <label className="flex items-center gap-2 text-sm text-slate-700">
        <input type="checkbox" checked={remote} onChange={(e) => setRemote(e.target.checked)}
               className="h-4 w-4" />
        Remote allowed
      </label>

      {error && <p role="alert" className="text-sm text-rose-700">{error}</p>}
      <button type="submit" disabled={busy}
              className="rounded-lg bg-indigo-600 px-4 py-2.5 font-semibold text-white hover:bg-indigo-700 disabled:opacity-50">
        {busy ? "Posting…" : "Post job"}
      </button>
    </form>
  );
}
