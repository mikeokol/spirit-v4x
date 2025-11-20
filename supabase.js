// supabase.js — Spirit v5.1
// --------------------------
import { createClient } from "@supabase/supabase-js";
import "dotenv/config";

const supabaseUrl = process.env.SUPABASE_URL;

// Prefer service role if available (full read/write)
// Fallback to anon for read-only operations.
const supabaseKey =
  process.env.SUPABASE_SERVICE_ROLE ||
  process.env.SUPABASE_ANON_KEY ||
  null;

if (!supabaseUrl || !supabaseKey) {
  console.warn("[Supabase] Missing URL or Key — check .env values");
}

const supabase = createClient(supabaseUrl, supabaseKey, {
  auth: {
    persistSession: false,
    autoRefreshToken: false,
  },
  global: {
    headers: {
      "X-Client-Info": "spirit-v5.1",
    },
  },
});

export default supabase;