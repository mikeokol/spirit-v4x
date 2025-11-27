// engine/memory.js — Spirit v7 Memory Engine
import { getSupabaseClient } from "../services/supabase.js";

const inMemoryStore = new Map();

export function getEmptyMemory(userId) {
  const now = new Date().toISOString();
  return {
    userId,
    version: "v7",
    profile: null,

    lastModes: [],
    lastTaskType: null,
    lastReplySummary: null,
    lastWorkoutPlan: null,
    lastCreatorScript: null,
    lastReflection: null,
    lastHybridPlan: null,

    live: {
      lastSessionId: null,
      lastStartedAt: null,
      lastEndedAt: null,
      lastCoachPush: null,
      totalSessions: 0,
    },

    liveHistory: [],

    createdAt: now,
    updatedAt: now,
  };
}

function mergeMemory(base, patch = {}) {
  const merged = {
    ...base,
    ...patch,
    live: { ...(base.live || {}), ...(patch.live || {}) },
  };

  // Trim history
  const hist = patch.liveHistory || base.liveHistory || [];
  merged.liveHistory = hist.slice(-50);

  merged.updatedAt = new Date().toISOString();
  return merged;
}

export async function loadUserMemory(userId) {
  const cached = inMemoryStore.get(userId);
  if (cached) return cached;

  const supabase = getSupabaseClient();

  if (supabase) {
    try {
      const { data, error } = await supabase
        .from("spirit_memory_v7")
        .select("memory")
        .eq("user_id", userId)
        .maybeSingle();

      if (data?.memory) {
        const merged = mergeMemory(getEmptyMemory(userId), data.memory);
        inMemoryStore.set(userId, merged);
        return merged;
      }
    } catch (_) {}
  }

  const fresh = getEmptyMemory(userId);
  inMemoryStore.set(userId, fresh);
  return fresh;
}

export async function saveUserMemory(userId, patch = {}) {
  const current = inMemoryStore.get(userId) || getEmptyMemory(userId);
  const merged = mergeMemory(current, patch);
  inMemoryStore.set(userId, merged);

  const supabase = getSupabaseClient();
  if (supabase) {
    try {
      await supabase.from("spirit_memory_v7").upsert({
        user_id: userId,
        memory: merged,
        updated_at: new Date().toISOString(),
      });
    } catch (_) {}
  }

  return merged;
}

export async function recordLiveEvent(userId, event) {
  const memory = await loadUserMemory(userId);
  const now = new Date().toISOString();

  const liveHistory = [...(memory.liveHistory || []), { ...event, ts: now }];

  let live = { ...(memory.live || {}) };

  if (event.type === "session_started") {
    live.lastSessionId = event.sessionId;
    live.lastStartedAt = now;
    live.totalSessions = (live.totalSessions || 0) + 1;
  }

  if (event.type === "session_ended") {
    live.lastEndedAt = now;
  }

  if (event.type === "coach_turn") {
    live.lastCoachPush = {
      ts: now,
      preview: event.reply?.slice(0, 140) || null,
    };
  }

  return saveUserMemory(userId, {
    live,
    liveHistory,
  });
}