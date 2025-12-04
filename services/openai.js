// services/supabase.js
// Spirit v7 — Safe Supabase Wrapper (never throws when keys missing)

import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY =
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY;

// ------------------------------------------------------
// If keys missing → run SAFELY in local-memory mode
// ------------------------------------------------------
if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.log("[supabase] No valid keys detected — using in-memory mode.");
  export const supabase = null;   // VERY IMPORTANT
  export const hasSupabase = false;
  return;
}

// ------------------------------------------------------
// If keys exist → initialize Supabase normally
// ------------------------------------------------------
export const supabase = createClient(SUPABASE_URL, SUPABASE_KEY, {
  auth: { persistSession: false },
});

export const hasSupabase = true;

console.log("[supabase] Client initialised.");