// engine/memory.js
// Spirit v7 — FULL MEMORY LAYER (ALL SYSTEMS SUPPORTED)

import { supabase, hasSupabase } from "../services/supabase.js";

const memoryStore = {
  users: {},
  liveSessions: {} // keyed by userId
};

// --------------------------------------------------------
// USER MEMORY — Load
// --------------------------------------------------------
export async function loadMemory(userId) {
  if (!hasSupabase) {
    return memoryStore.users[userId] || {};
  }

  const { data, error } = await supabase
    .from("spirit_memory")
    .select("data")
    .eq("userId", userId)
    .single();

  if (error || !data) return {};
  return data.data || {};
}

// --------------------------------------------------------
// USER MEMORY — Save
// --------------------------------------------------------
export async function saveMemory(userId, newData) {
  if (!hasSupabase) {
    memoryStore.users[userId] = {
      ...(memoryStore.users[userId] || {}),
      ...newData,
    };
    return;
  }

  await supabase.from("spirit_memory").upsert({
    userId,
    data: newData,
  });
}

// --------------------------------------------------------
// CREATOR HISTORY
// --------------------------------------------------------
export async function saveCreatorHistory(userId, entry) {
  const current = await loadMemory(userId);
  const history = current.creatorHistory || [];

  history.push({ ts: Date.now(), ...entry });

  await saveMemory(userId, { ...current, creatorHistory: history });
}

// --------------------------------------------------------
// HYBRID HISTORY
// --------------------------------------------------------
export async function saveHybridHistory(userId, entry) {
  const current = await loadMemory(userId);
  const history = current.hybridHistory || [];

  history.push({ ts: Date.now(), ...entry });

  await saveMemory(userId, { ...current, hybridHistory: history });
}

// --------------------------------------------------------
// REFLECTION HISTORY
// --------------------------------------------------------
export async function saveReflectionHistory(userId, reflection) {
  const current = await loadMemory(userId);
  const history = current.reflections || [];

  history.push({ ts: Date.now(), reflection });

  await saveMemory(userId, { ...current, reflections: history });
}

// --------------------------------------------------------
// FITNESS PLAN SAVE
// --------------------------------------------------------
export async function saveFitnessPlan(userId, plan) {
  const current = await loadMemory(userId);
  await saveMemory(userId, { ...current, lastFitnessPlan: plan });
}

// --------------------------------------------------------
// LIVE SESSIONS — SAVE EVENT (Local + Supabase)
// --------------------------------------------------------
export async function recordLiveEvent(userId, sessionId, eventData) {
  const entry = {
    ts: new Date().toISOString(),
    userId,
    sessionId,
    ...eventData,
  };

  // local
  if (!memoryStore.liveSessions[userId]) {
    memoryStore.liveSessions[userId] = [];
  }
  memoryStore.liveSessions[userId].push(entry);

  // supabase
  if (hasSupabase) {
    const { error } = await supabase.from("live_sessions").insert({
      userid: userId,
      sessionid: sessionId,
      event: entry,
      ts: entry.ts,
    });

    if (error) console.error("[SUPABASE LIVE INSERT ERROR]", error);
  }
}

// --------------------------------------------------------
// LIVE SESSIONS — FETCH HISTORY
// --------------------------------------------------------
export async function getUserLiveHistory(userId) {
  if (hasSupabase) {
    const { data, error } = await supabase
      .from("live_sessions")
      .select("*")
      .eq("userid", userId)
      .order("ts", { ascending: true });

    if (!error && data) return data;
  }

  return memoryStore.liveSessions[userId] || [];
}

// --------------------------------------------------------
// LIVE SESSIONS — CLEAR HISTORY
// --------------------------------------------------------
export async function clearLiveHistory(userId) {
  delete memoryStore.liveSessions[userId];

  if (hasSupabase) {
    await supabase.from("live_sessions").delete().eq("userid", userId);
  }
}