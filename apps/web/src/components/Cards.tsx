import type { CandidateCard, JobCard } from "../lib/api";

function ScoreBadge({ score }: { score: number }) {
  const tone =
    score >= 70 ? "bg-emerald-100 text-emerald-800"
      : score >= 45 ? "bg-amber-100 text-amber-800"
        : "bg-slate-100 text-slate-700";
  return (
    <span className={`rounded-full px-3 py-1 text-sm font-semibold ${tone}`}>
      {Math.round(score)}% fit
    </span>
  );
}

function Reasons({ reasons }: { reasons: string[] }) {
  if (!reasons.length) return null;
  return (
    <ul className="mt-3 space-y-1">
      {reasons.map((reason) => (
        <li key={reason} className="flex gap-2 text-sm text-slate-600">
          <span aria-hidden="true" className="text-indigo-400">•</span>
          {reason}
        </li>
      ))}
    </ul>
  );
}

export function salaryLabel(job: JobCard): string | null {
  if (!job.normalized_salary) return null;
  const value = job.normalized_salary;
  const formatted =
    value >= 1000
      ? `$${Math.round(value / 1000)}k`
      : `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  return `${formatted}${job.pay_period && job.pay_period !== "YEARLY" ? ` / ${job.pay_period.toLowerCase()}` : "/yr"}`;
}

export function JobCardBody({ job }: { job: JobCard }) {
  const salary = salaryLabel(job);
  return (
    <div className="flex h-full flex-col p-6">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="truncate text-xl font-bold text-slate-900">{job.title}</h2>
          <p className="truncate text-slate-600">{job.company_name}</p>
        </div>
        <ScoreBadge score={job.score} />
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {job.location && <Tag>{job.location}</Tag>}
        {job.remote_allowed && <Tag tone="indigo">Remote</Tag>}
        {job.formatted_work_type && <Tag>{job.formatted_work_type}</Tag>}
        {job.formatted_experience_level && <Tag>{job.formatted_experience_level}</Tag>}
        {salary && <Tag tone="emerald">{salary}</Tag>}
      </div>

      <Reasons reasons={job.reasons} />

      <p className="mt-4 flex-1 overflow-hidden text-sm leading-relaxed text-slate-600">
        {job.description.slice(0, 520)}
        {job.description.length > 520 ? "…" : ""}
      </p>

      {/* USAJOBS terms require crediting the source and sending applicants to the
          official announcement, so the attribution is part of the card, not optional. */}
      {job.source === "usajobs" && (
        <p className="mt-2 text-xs text-slate-400">
          Source: USAJOBS. Apply on the official announcement. Not endorsed by OPM.
        </p>
      )}
    </div>
  );
}

export function CandidateCardBody({ candidate }: { candidate: CandidateCard }) {
  return (
    <div className="flex h-full flex-col p-6">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="truncate text-xl font-bold text-slate-900">
            {candidate.display_name}
          </h2>
          {candidate.headline && (
            <p className="truncate text-slate-600">{candidate.headline}</p>
          )}
        </div>
        <ScoreBadge score={candidate.score} />
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {candidate.predicted_category && <Tag>{candidate.predicted_category}</Tag>}
        {candidate.years_experience != null && (
          <Tag>~{Math.round(candidate.years_experience)} yrs</Tag>
        )}
        {candidate.is_demo && <Tag tone="amber">Sample profile</Tag>}
      </div>

      <Reasons reasons={candidate.reasons} />

      <p className="mt-4 flex-1 overflow-hidden text-sm leading-relaxed text-slate-600">
        {candidate.resume_preview}…
      </p>

      {candidate.is_demo && (
        <p className="mt-2 text-xs text-slate-400">
          Sample profile from the training corpus — useful for testing ranking, but it
          cannot match back.
        </p>
      )}
    </div>
  );
}

function Tag({
  children,
  tone = "slate",
}: {
  children: React.ReactNode;
  tone?: "slate" | "indigo" | "emerald" | "amber";
}) {
  const tones = {
    slate: "bg-slate-100 text-slate-700",
    indigo: "bg-indigo-100 text-indigo-700",
    emerald: "bg-emerald-100 text-emerald-700",
    amber: "bg-amber-100 text-amber-800",
  };
  return <span className={`rounded-md px-2 py-1 text-xs font-medium ${tones[tone]}`}>{children}</span>;
}
