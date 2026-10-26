import { createClient } from "@supabase/supabase-js";

const url = import.meta.env.VITE_SUPABASE_URL as string;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string;

if (!url || !anonKey) {
  // Surfaced loudly rather than failing later with an opaque network error.
  console.error(
    "Missing VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY. Copy apps/web/.env.example to .env.local."
  );
}

export const supabase = createClient(url ?? "", anonKey ?? "", {
  auth: { persistSession: true, autoRefreshToken: true },
});
