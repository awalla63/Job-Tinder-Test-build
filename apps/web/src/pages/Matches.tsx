import { useEffect, useState } from "react";
import { api, type MatchRow } from "../lib/api";

export default function Matches() {
  const [rows, setRows] = useState<MatchRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.matches()
      .then(setRows)
      .catch((e) => setError((e as Error).message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      <h1 className="mb-6 text-2xl font-bold text-slate-900">Your matches</h1>

      {loading && <p className="text-slate-600">Loading…</p>}
      {error && <p role="alert" className="text-rose-700">{error}</p>}

      {!loading && !error && rows.length === 0 && (
        <p className="rounded-xl border border-dashed border-slate-300 p-8 text-center text-slate-500">
          No matches yet. Like a few jobs and they'll show up here.
        </p>
      )}

      <ul className="space-y-3">
        {rows.map((row) => (
          <li
            key={row.id}
            className="flex items-center justify-between gap-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
          >
            <div className="min-w-0">
              <p className="truncate font-semibold text-slate-900">{row.job_title}</p>
              <p className="truncate text-sm text-slate-600">{row.company_name}</p>
              <p className="mt-1 text-xs text-slate-400">
                {new Date(row.created_at).toLocaleDateString()}
                {row.source === "usajobs" && " · Source: USAJOBS"}
              </p>
            </div>
            {row.application_url ? (
              <a
                href={row.application_url}
                target="_blank"
                rel="noreferrer noopener"
                className="shrink-0 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700 focus-visible:ring-4 focus-visible:ring-indigo-200"
              >
                Apply
              </a>
            ) : (
              <span className="shrink-0 text-sm text-slate-400">No link</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
