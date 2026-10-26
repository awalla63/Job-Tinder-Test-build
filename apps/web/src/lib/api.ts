import { supabase } from "./supabase";

const BASE = (import.meta.env.VITE_API_URL as string) ?? "http://localhost:8000";

export type JobCard = {
  id: string;
  title: string;
  company_name: string;
  location: string | null;
  description: string;
  normalized_salary: number | null;
  pay_period: string | null;
  formatted_work_type: string | null;
  formatted_experience_level: string | null;
  remote_allowed: boolean;
  application_url: string | null;
  source: string;
  score: number;
  reasons: string[];
};

export type CandidateCard = {
  id: string;
  display_name: string;
  headline: string | null;
  resume_preview: string;
  years_experience: number | null;
  predicted_category: string | null;
  is_demo: boolean;
  score: number;
  reasons: string[];
};

export type SwipeResult = {
  recorded: boolean;
  matched: boolean;
  match_id: string | null;
  application_url: string | null;
};

export type MatchRow = {
  id: string;
  job_id: string;
  job_title: string;
  company_name: string;
  application_url: string | null;
  source: string;
  counterpart_name: string | null;
  created_at: string;
};

export type Profile = {
  onboarded: boolean;
  role: "seeker" | "employer" | null;
  display_name?: string;
  has_survey?: boolean;
  has_resume?: boolean;
  company_name?: string | null;
  job_count?: number;
};

async function authHeaders(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = {
    ...(await authHeaders()),
    ...(init.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
    ...(init.headers ?? {}),
  };
  const response = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

export const api = {
  me: () => request<Profile>("/me"),

  completeProfile: (body: {
    role: "seeker" | "employer";
    display_name: string;
    company_name?: string;
  }) => request<{ ok: boolean }>("/auth/complete-profile", {
    method: "POST",
    body: JSON.stringify(body),
  }),

  uploadResume: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ ok: boolean; characters: number; years_experience: number | null; preview: string }>(
      "/seeker/resume",
      { method: "POST", body: form }
    );
  },

  saveSurvey: (body: Record<string, unknown>) =>
    request<{ ok: boolean }>("/survey", { method: "POST", body: JSON.stringify(body) }),

  jobDeck: (limit = 20) => request<JobCard[]>(`/deck/jobs?limit=${limit}`),

  candidateDeck: (jobId: string, limit = 20) =>
    request<CandidateCard[]>(`/deck/candidates?job_id=${encodeURIComponent(jobId)}&limit=${limit}`),

  swipe: (body: {
    target_type: "job" | "seeker";
    target_id: string;
    direction: "like" | "pass";
    score_at_swipe?: number;
  }) => request<SwipeResult>("/swipe", { method: "POST", body: JSON.stringify(body) }),

  matches: () => request<MatchRow[]>("/matches"),

  employerJobs: () =>
    request<Array<{ id: string; title: string; company_name: string; location: string | null; likes: number; created_at: string }>>(
      "/employer/jobs"
    ),

  createJob: (body: Record<string, unknown>) =>
    request<{ ok: boolean; job_id: string }>("/employer/jobs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
