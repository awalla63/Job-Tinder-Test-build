import { useCallback, useEffect, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import type { Session } from "@supabase/supabase-js";
import { supabase } from "./lib/supabase";
import { api, type Profile } from "./lib/api";
import Login from "./pages/Login";
import Onboarding from "./pages/Onboarding";
import Swipe from "./pages/Swipe";
import Matches from "./pages/Matches";
import EmployerJobs from "./pages/EmployerJobs";

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => setSession(data.session));
    const { data: sub } = supabase.auth.onAuthStateChange((_event, next) => {
      setSession(next);
      setProfile(null);
    });
    return () => sub.subscription.unsubscribe();
  }, []);

  const refreshProfile = useCallback(async () => {
    if (!session) {
      setProfile(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      setProfile(await api.me());
    } catch {
      setProfile({ onboarded: false, role: null });
    } finally {
      setLoading(false);
    }
  }, [session]);

  useEffect(() => {
    void refreshProfile();
  }, [refreshProfile]);

  if (!session) return <Login />;
  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center text-slate-600">
        Loading…
      </div>
    );
  }

  const needsOnboarding =
    !profile?.onboarded ||
    !profile.has_survey ||
    (profile.role === "seeker" && !profile.has_resume);

  return (
    <div className="min-h-screen bg-slate-50">
      <Nav role={profile?.role ?? null} />
      <main>
        {needsOnboarding ? (
          <Onboarding profile={profile ?? { onboarded: false, role: null }} onDone={refreshProfile} />
        ) : (
          <Routes>
            <Route
              path="/"
              element={<Navigate to={profile?.role === "employer" ? "/employer/jobs" : "/swipe"} replace />}
            />
            <Route path="/swipe" element={<Swipe />} />
            <Route path="/matches" element={<Matches />} />
            <Route path="/employer/jobs" element={<EmployerJobs />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        )}
      </main>
    </div>
  );
}

function Nav({ role }: { role: "seeker" | "employer" | null }) {
  const { pathname } = useLocation();
  const links =
    role === "employer"
      ? [{ to: "/employer/jobs", label: "Postings" }, { to: "/matches", label: "Matches" }]
      : [{ to: "/swipe", label: "Swipe" }, { to: "/matches", label: "Matches" }];

  return (
    <header className="border-b border-slate-200 bg-white">
      <nav className="mx-auto flex max-w-4xl items-center justify-between px-4 py-3">
        <span className="font-bold text-indigo-600">Job Tinder</span>
        <div className="flex items-center gap-1">
          {links.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              aria-current={pathname === link.to ? "page" : undefined}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                pathname === link.to
                  ? "bg-indigo-50 text-indigo-700"
                  : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              {link.label}
            </Link>
          ))}
          <button
            onClick={() => supabase.auth.signOut()}
            className="ml-2 rounded-lg px-3 py-1.5 text-sm text-slate-500 hover:bg-slate-100"
          >
            Sign out
          </button>
        </div>
      </nav>
    </header>
  );
}
