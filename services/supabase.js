// services/supabase.js
// Final Stable Version — Spirit v7

import { createClient } from "@supabase/supabase-js";

const url = process.env.SUPABASE_URL;
const key =
  process.env.SUPABASE_SERVICE_ROLE_KEY ||
  process.env.SUPABASE_ANON_KEY;

// ----------------------------------------------
// Detect if Supabase is available
// ----------------------------------------------
export const hasSupabase = Boolean(url && key);

// ----------------------------------------------
// Create client (or fake one)
// ----------------------------------------------
export const supabase = hasSupabase
  ? createClient(url, key)
  : {
      // Fake client to prevent crashes in local mode
      from() {
        return {
          select() {
            return { data: null, error: null };
          },
          insert() {
            return { data: null, error: null };
          },
          upsert() {
            return { data: null, error: null };
          },
          update() {
            return { data: null, error: null };
          },
          eq() {
            return this;
          },
          single() {
            return { data: null, error: null };
          },
        };
      },
    };

console.log(
  hasSupabase
    ? "[supabase] Client initialised"
    : "[supabase] No valid env keys — using local in-memory mode"
);