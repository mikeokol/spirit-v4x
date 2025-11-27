// services/supabase.js
import { createClient } from "@supabase/supabase-js";

let supabase = null;

const url = process.env.SUPABASE_URL;
const key =
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY;

if (!url || !key) {
  console.warn(
    "[supabase] Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY/ANON_KEY. " +
      "Spirit v7 will use in-memory-only memory. Fine for dev; configure Supabase for persistence."
  );
} else {
  supabase = createClient(url, key, {
    auth: {
      persistSession: false,
    },
  });
  console.log("[supabase] Client initialised.");
}

export function getSupabaseClient() {
  return supabase;
}