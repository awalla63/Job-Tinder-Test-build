import { useCallback, useEffect, useState } from "react";
import { api, type JobCard } from "../lib/api";
import { JobCardBody } from "../components/Cards";
import { SwipeDeck, type Decision } from "../components/SwipeDeck";

export default function Swipe() {
  const [cards, setCards] = useState<JobCard[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [match, setMatch] = useState<{ title: string; url: string | null } | null>(null);
  const [decided, setDecided] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setCards(await api.jobDeck(20));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Fetch the next page once the user is near the end, so swiping never stalls on
  // a network round trip.
  useEffect(() => {
    if (cards.length && decided >= cards.length - 5) {
      api.jobDeck(20)
        .then((next) => {
          const seen = new Set(cards.map((c) => c.id));
          const fresh = next.filter((c) => !seen.has(c.id));
          if (fresh.length) setCards((current) => [...current, ...fresh]);
        })
        .catch(() => {
          /* prefetch is best-effort */
        });
    }
  }, [decided, cards]);

  const onDecide = useCallback(async (job: JobCard, decision: Decision) => {
    setDecided((n) => n + 1);
    try {
      const result = await api.swipe({
        target_type: "job",
        target_id: job.id,
        direction: decision,
        score_at_swipe: job.score,
      });
      if (result.matched) {
        setMatch({ title: job.title, url: result.application_url });
      }
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  if (loading) return <Centered>Finding jobs that fit…</Centered>;

  if (error) {
    return (
      <Centered>
        <p role="alert" className="text-rose-700">{error}</p>
        <button onClick={load} className="mt-4 rounded-lg bg-indigo-600 px-4 py-2 font-medium text-white">
          Try again
        </button>
      </Centered>
    );
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      <h1 className="mb-6 text-2xl font-bold text-slate-900">Jobs for you</h1>

      <SwipeDeck
        items={cards}
        getKey={(job) => job.id}
        describe={(job) =>
          `${job.title} at ${job.company_name}, ${Math.round(job.score)} percent fit`
        }
        renderCard={(job) => <JobCardBody job={job} />}
        onDecide={onDecide}
        emptyState={
          <div>
            <p className="font-medium text-slate-700">That's everything for now.</p>
            <p className="mt-1 text-sm text-slate-500">
              Loosen your filters or check back after the next job refresh.
            </p>
            <button onClick={load} className="mt-4 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white">
              Reload deck
            </button>
          </div>
        }
      />

      {match && (
        <div
          role="alertdialog"
          aria-modal="true"
          aria-labelledby="match-title"
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 px-4"
        >
          <div className="w-full max-w-sm rounded-2xl bg-white p-6 text-center shadow-xl">
            <h2 id="match-title" className="text-xl font-bold text-slate-900">It's a match</h2>
            <p className="mt-2 text-slate-600">{match.title}</p>
            <div className="mt-6 flex flex-col gap-2">
              {match.url && (
                <a
                  href={match.url}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="rounded-lg bg-indigo-600 px-4 py-2.5 font-semibold text-white hover:bg-indigo-700"
                >
                  Apply now
                </a>
              )}
              <button
                onClick={() => setMatch(null)}
                autoFocus
                className="rounded-lg border border-slate-300 px-4 py-2.5 font-medium text-slate-700 hover:bg-slate-50"
              >
                Keep swiping
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center px-4 text-center text-slate-600">
      {children}
    </div>
  );
}
