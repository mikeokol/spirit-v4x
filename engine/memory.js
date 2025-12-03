// engine/memory.js
import { getSupabaseClient } from "../services/supabase.js";

// In-memory cache as a safe fallback and for faster reads.
const inMemoryStore = new Map();

export function getEmptyMemory(userId) {
  const now = new Date().toISOString();
  return {
    userId,
    version: "v7.1",

    // High-level profile / identity
    profile: null, // e.g. free-form profile / notes

    // Preference flags (can be set from Sanctuary / onboarding later)
    preferences: {
      allowMemory: true,     // later can be toggled from UI
      name: null,
      liveTone: "elite",     // e.g. "elite", "gentle"
      fitnessGoal: null,     // "fat_loss" | "muscle_gain" | "performance" | "general"
    },

    // Fitness-specific memory
    fitness: {
      currentProgram: null,      // e.g. { type, split, daysPerWeek }
      lastGoal: null,            // "fat_loss" | "muscle_gain" | ...
      lastRPE: null,             // numeric, last reported difficulty
      lastWorkoutSummary: null,  // short text summary
    },

    // Generic engine traces
    lastModes: [],
    lastTaskType: null,
    lastReplySummary: null,
    lastWorkoutPlan: null,
    lastCreatorScript: null,
    lastReflection: null,
    lastHybridPlan: null,

    // Live coaching specific state
    live: {
      lastSessionId: null,
      lastStartedAt: null,
      lastEndedAt: null,
      lastCoachPush: null, // { ts, preview }
      totalSessions: 0,

      // Stateful "where am I in the live flow?"
      // Example: { phase, goal, block, step, expecting }
      state: null,
    },

    // Rolling log of recent live events (trimmed)
    liveHistory: [],

    createdAt: now,
    updatedAt: now,
  };
}

function mergeMemory(base, patch = {}) {
  const merged = {
    ...base,
    ...patch,

    // deep-merge preference object
    preferences: {
      ...(base.preferences || {}),
      ...(patch.preferences || {}),
    },

    // deep-merge fitness object
    fitness: {
      ...(base.fitness || {}),
      ...(patch.fitness || {}),
    },

    // deep-merge live object
    live: {
      ...(base.live || {}),
      ...(patch.live || {}),
    },
  };

  // Clamp liveHistory length if present
  if (patch.liveHistory || base.liveHistory) {
    const hist = patch.liveHistory || base.liveHistory || [];
    merged.liveHistory = hist.slice(-50);
  }

  merged.updatedAt = new Date().toISOString();
  return merged;
}

// Load memory for a user from Supabase (if configured) with in-memory fallback.
export async function loadUserMemory(userId) {
  console.log("[MEMORY] loadUserMemory:", { userId });

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

      if (error) {
        console.warn("[memory] Supabase load error:", error.message);
      } else if (data && data.memory) {
        const base = getEmptyMemory(userId);
        const merged = mergeMemory(base, data.memory);
        inMemoryStore.set(userId, merged);
        return merged;
      }
    } catch (err) {
      console.warn("[memory] Supabase load exception:", err.message || err);
    }
  }

  const fresh = getEmptyMemory(userId);
  inMemoryStore.set(userId, fresh);
  return fresh;
}

// Save memory for a user to Supabase (if configured) and update in-memory cache.
export async function saveUserMemory(userId, patch = {}) {
  const current = inMemoryStore.get(userId) || getEmptyMemory(userId);
  const merged = mergeMemory(current, patch);
  inMemoryStore.set(userId, merged);

  const supabase = getSupabaseClient();
  if (supabase) {
    try {
      const payload = {
        user_id: userId,
        memory: merged,
        updated_at: new Date().toISOString(),
      };

      const { error } = await supabase
        .from("spirit_memory_v7")
        .upsert(payload, {
          onConflict: "user_id",
        });

      if (error) {
        console.warn("[memory] Supabase save error:", error.message);
      }
    } catch (err) {
      console.warn("[memory] Supabase save exception:", err.message || err);
    }
  }

  return merged;
}

// Convenience: record a live coaching event and update counters.
export async function recordLiveEvent(userId, event) {
  const memory = await loadUserMemory(userId);
  const now = new Date().toISOString();

  const liveHistory = Array.isArray(memory.liveHistory)
    ? [...memory.liveHistory]
    : [];

  liveHistory.push({
    ...event,
    ts: now,
  });

  let live = {
    ...(memory.live || {}),
  };

  if (event.type === "session_started") {
    live.lastSessionId = event.sessionId;
    live.lastStartedAt = now;
    live.totalSessions = (live.totalSessions || 0) + 1;
    // reset state when a new live session begins
    live.state = null;
  }

  if (event.type === "session_ended") {
    live.lastEndedAt = now;
    // we *don’t* clear state here yet; that happens inside live coach logic if desired
  }

  if (event.type === "coach_turn") {
    live.lastCoachPush = {
      ts: now,
      preview:
        typeof event.reply === "string" ? event.reply.slice(0, 140) : null,
    };
  }

  return saveUserMemory(userId, {
    live,
    liveHistory,
  });
}