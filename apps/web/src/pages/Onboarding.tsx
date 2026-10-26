import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type Profile } from "../lib/api";

const WORK_TYPES = ["Full-time", "Part-time", "Contract", "Internship", "Temporary"];
const EXPERIENCE_LEVELS = [
  "Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive",
];

export default function Onboarding({ profile, onDone }: { profile: Profile; onDone: () => void }) {
  const navigate = useNavigate();
  const [step, setStep] = useState<"role" | "upload" | "survey">(
    !profile.onboarded ? "role" : profile.role === "seeker" && !profile.has_resume ? "upload" : "survey"
  );
  const [role, setRole] = useState<"seeker" | "employer">(profile.role ?? "seeker");
  const [displayName, setDisplayName] = useState(profile.display_name ?? "");
  const [companyName, setCompanyName] = useState(profile.company_name ?? "");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function saveRole(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.completeProfile({
        role,
        display_name: displayName,
        company_name: role === "employer" ? companyName : undefined,
      });
      setStep(role === "seeker" ? "upload" : "survey");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-xl px-4 py-12">
      <h1 className="text-2xl font-bold text-slate-900">Set up your account</h1>

      {error && (
        <p role="alert" className="mt-4 rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700">
          {error}
        </p>
      )}

      {step === "role" && (
        <form onSubmit={saveRole} className="mt-6 space-y-5">
          <fieldset>
            <legend className="text-sm font-medium text-slate-700">I am a…</legend>
            <div className="mt-2 grid grid-cols-2 gap-3">
              {(["seeker", "employer"] as const).map((option) => (
                <label
                  key={option}
                  className={`cursor-pointer rounded-xl border-2 p-4 text-center font-medium transition ${
                    role === option
                      ? "border-indigo-500 bg-indigo-50 text-indigo-700"
                      : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
                  }`}
                >
                  <input
                    type="radio"
                    name="role"
                    value={option}
                    checked={role === option}
                    onChange={() => setRole(option)}
                    className="sr-only"
                  />
                  {option === "seeker" ? "Job seeker" : "Employer"}
                </label>
              ))}
            </div>
          </fieldset>

          <Field label="Your name" value={displayName} onChange={setDisplayName} required />
          {role === "employer" && (
            <Field label="Company name" value={companyName} onChange={setCompanyName} required />
          )}

          <PrimaryButton busy={busy}>Continue</PrimaryButton>
        </form>
      )}

      {step === "upload" && <ResumeStep onDone={() => setStep("survey")} />}

      {step === "survey" && (
        <SurveyStep
          role={role}
          onDone={() => {
            onDone();
            navigate(role === "seeker" ? "/swipe" : "/employer/jobs");
          }}
        />
      )}
    </div>
  );
}

function ResumeStep({ onDone }: { onDone: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ characters: number; years_experience: number | null } | null>(null);

  async function upload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const response = await api.uploadResume(file);
      setResult(response);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-6 space-y-5">
      <div>
        <label htmlFor="resume" className="block text-sm font-medium text-slate-700">
          Upload your resume (PDF, DOCX or TXT)
        </label>
        <input
          id="resume"
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={upload}
          disabled={busy}
          className="mt-2 block w-full rounded-lg border border-slate-300 px-3 py-2 text-sm file:mr-3 file:rounded-md file:border-0 file:bg-indigo-50 file:px-3 file:py-1.5 file:text-indigo-700"
        />
        <p className="mt-2 text-xs text-slate-500">
          We read the text to match you against postings. Nothing is shared with employers
          until you both like each other.
        </p>
      </div>

      {busy && <p role="status" className="text-sm text-slate-600">Reading your resume…</p>}
      {error && (
        <p role="alert" className="rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</p>
      )}
      {result && (
        <p role="status" className="rounded-lg bg-emerald-50 px-4 py-3 text-sm text-emerald-800">
          Read {result.characters.toLocaleString()} characters
          {result.years_experience != null && `, about ${Math.round(result.years_experience)} years of history`}.
        </p>
      )}

      <button
        type="button"
        onClick={onDone}
        disabled={!result}
        className="w-full rounded-lg bg-indigo-600 px-4 py-2.5 font-semibold text-white transition hover:bg-indigo-700 disabled:opacity-40"
      >
        Continue
      </button>
    </div>
  );
}

function SurveyStep({ role, onDone }: { role: "seeker" | "employer"; onDone: () => void }) {
  const [minSalary, setMinSalary] = useState("");
  const [remotePref, setRemotePref] = useState<"remote" | "onsite" | "any">("any");
  const [workTypes, setWorkTypes] = useState<string[]>([]);
  const [levels, setLevels] = useState<string[]>([]);
  const [states, setStates] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function toggle(list: string[], setList: (v: string[]) => void, value: string) {
    setList(list.includes(value) ? list.filter((v) => v !== value) : [...list, value]);
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.saveSurvey({
        min_salary: minSalary ? Number(minSalary) : null,
        remote_pref: remotePref,
        work_types: workTypes.length ? workTypes : null,
        experience_levels: levels.length ? levels : null,
        states: states
          ? states.split(",").map((s) => s.trim().toUpperCase()).filter(Boolean)
          : null,
        answers: { role },
      });
      onDone();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="mt-6 space-y-6">
      <p className="text-sm text-slate-600">
        These are filters, not preferences we score on — leave anything blank to keep it open.
      </p>

      <Field
        label="Minimum salary (USD/year)"
        value={minSalary}
        onChange={setMinSalary}
        type="number"
        hint="Postings that don't publish a salary are always shown."
      />

      <fieldset>
        <legend className="text-sm font-medium text-slate-700">Work location</legend>
        <div className="mt-2 flex gap-3">
          {(["any", "remote", "onsite"] as const).map((option) => (
            <label key={option} className="flex items-center gap-2 text-sm text-slate-700">
              <input
                type="radio"
                name="remote"
                checked={remotePref === option}
                onChange={() => setRemotePref(option)}
                className="h-4 w-4"
              />
              {option === "any" ? "No preference" : option === "remote" ? "Remote only" : "On-site only"}
            </label>
          ))}
        </div>
      </fieldset>

      <CheckGroup legend="Work types" options={WORK_TYPES} selected={workTypes}
                  onToggle={(v) => toggle(workTypes, setWorkTypes, v)} />
      <CheckGroup legend="Experience levels" options={EXPERIENCE_LEVELS} selected={levels}
                  onToggle={(v) => toggle(levels, setLevels, v)} />

      <Field label="States (comma-separated, e.g. NC, NY)" value={states} onChange={setStates} />

      {error && (
        <p role="alert" className="rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</p>
      )}
      <PrimaryButton busy={busy}>Finish</PrimaryButton>
    </form>
  );
}

function CheckGroup({
  legend, options, selected, onToggle,
}: { legend: string; options: string[]; selected: string[]; onToggle: (v: string) => void }) {
  return (
    <fieldset>
      <legend className="text-sm font-medium text-slate-700">{legend}</legend>
      <div className="mt-2 flex flex-wrap gap-2">
        {options.map((option) => (
          <label
            key={option}
            className={`cursor-pointer rounded-lg border px-3 py-1.5 text-sm transition ${
              selected.includes(option)
                ? "border-indigo-500 bg-indigo-50 text-indigo-700"
                : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
            }`}
          >
            <input
              type="checkbox"
              checked={selected.includes(option)}
              onChange={() => onToggle(option)}
              className="sr-only"
            />
            {option}
          </label>
        ))}
      </div>
    </fieldset>
  );
}

function Field({
  label, value, onChange, type = "text", required, hint,
}: {
  label: string; value: string; onChange: (v: string) => void;
  type?: string; required?: boolean; hint?: string;
}) {
  const id = label.toLowerCase().replace(/[^a-z]+/g, "-");
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium text-slate-700">{label}</label>
      <input
        id={id}
        type={type}
        required={required}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 focus:outline-none"
      />
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  );
}

function PrimaryButton({ busy, children }: { busy: boolean; children: React.ReactNode }) {
  return (
    <button
      type="submit"
      disabled={busy}
      className="w-full rounded-lg bg-indigo-600 px-4 py-2.5 font-semibold text-white transition hover:bg-indigo-700 focus-visible:ring-4 focus-visible:ring-indigo-200 disabled:opacity-50"
    >
      {busy ? "Saving…" : children}
    </button>
  );
}
