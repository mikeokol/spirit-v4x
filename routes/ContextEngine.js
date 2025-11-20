// routes/ContextEngine.js
// ---------------------------------------------------------------------------
// Spirit v5.1 — Context Engine
// Handles:
//  • Reading last intention / reflection / mode
//  • Storing reflections
//  • Storing fitness training blocks in sessions
// ---------------------------------------------------------------------------

import supabase from "../supabase.js";

// ---------------------------------------------------------------------------
// Get last known context for a user
// ---------------------------------------------------------------------------
export async function getLastContext(userId) {
  if (!userId) {
    return { lastIntention: null, lastReflection: null, lastMode: null };
  }

  const { data: lastReflectionRow, error: reflectionError } = await supabase
    .from("reflections")
    .select("intention, mode")
    .eq("user_id", userId)
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (reflectionError) {
    console.warn("[ContextEngine] reflections query error:", reflectionError.message);
  }

  const { data: sessionRow, error: sessionError } = await supabase
    .from("sessions")
    .select("last_intention, last_mode")
    .eq("user_id", userId)
    .maybeSingle();

  if (sessionError) {
    console.warn("[ContextEngine] sessions query error:", sessionError.message);
  }

  return {
    lastIntention: sessionRow?.last_intention || null,
    lastReflection: lastReflectionRow?.intention || null,
    lastMode: sessionRow?.last_mode || lastReflectionRow?.mode || null,
  };
}

// ---------------------------------------------------------------------------
// Store a reflection + update session last_intention / last_mode
// ---------------------------------------------------------------------------
export async function storeReflection({ userId, prompt, mode, reply }) {
  if (!userId) return;

  const now = new Date().toISOString();

  const reflectionPayload = {
    user_id: userId,
    intention: prompt,
    mode,
    content: JSON.stringify({ prompt, reply, mode }),
    created_at: now,
  };

  const { error: reflectionError } = await supabase
    .from("reflections")
    .insert(reflectionPayload);

  if (reflectionError) {
    console.error("[ContextEngine] storeReflection insert error:", reflectionError.message);
  }

  const { error: sessionError } = await supabase.from("sessions").upsert(
    {
      user_id: userId,
      last_intention: prompt,
      last_mode: mode,
      updated_at: now,
    },
    { onConflict: "user_id" }
  );

  if (sessionError) {
    console.error("[ContextEngine] storeReflection upsert session error:", sessionError.message);
  }
}

// ---------------------------------------------------------------------------
// Store fitness training block into sessions.training_block (with workouts)
// ---------------------------------------------------------------------------
export async function storeTrainingBlock({ userId, planText, meta = {}, workouts }) {
  if (!userId || !planText) return;

  const now = new Date().toISOString();

  const daysValue =
    meta.days !== undefined && meta.days !== null
      ? Number(meta.days) || meta.days
      : null;

  const blockPayload = {
    plan_text: planText,
    goal: meta.goal || null,
    specific_goal: meta.specificGoal || null,
    experience: meta.experience || null,
    days: daysValue,
    gender: meta.gender || "unspecified",
    weight: meta.weight || null,
    height: meta.height || null,
    workouts: Array.isArray(workouts) ? workouts : [],
    created_at: now,
  };

  const upsertPayload = {
    user_id: userId,
    training_block: blockPayload,
    training_day: 1,
    difficulty_adjustment: "normal",
    last_session_completed_at: null,
    last_mode: "body",
    gender: blockPayload.gender,
    updated_at: now,
  };

  const { error } = await supabase
    .from("sessions")
    .upsert(upsertPayload, { onConflict: "user_id" });

  if (error) {
    console.error("[ContextEngine] storeTrainingBlock upsert error:", error.message);
  }
}