// supabase.js — Spirit v4.x
// --------------------------
import { createClient } from "@supabase/supabase-js";
import "dotenv/config";

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.warn("[Supabase] Missing SUPABASE_URL or SUPABASE_KEY env vars");
}

const supabase = createClient(supabaseUrl, supabaseKey, {
  auth: { persistSession: false },
  global: { headers: { "X-Client-Info": "spirit-v4.x" } },
});

export default supabase;