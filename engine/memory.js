// memory.js — Trinity v7 Cognitive Engine Memory Module

import { supabase } from "../services/supabase.js";

// ================================
// READ MEMORY
// ================================
export async function pullUserMemory(userId) {
  try {
    // Load reflections newest → oldest
    const { data: reflections = [] } = await supabase
      .from("reflections")
      .select("*")
      .eq("userId", userId)
      .order("created_at", { ascending: false });

    // Load session history (fitness/live)
    const { data: sessions = [] } = await supabase
      .from("sessions")
      .select("*")
      .eq("userId", userId)
      .order("created_at", { ascending: false });

    // Load creator history
    const { data: creator = [] } = await supabase
      .from("creator_logs")
      .select("*")
      .eq("userId", userId)
      .order("created_at", { ascending: false });

    // Unified memory snapshot
    return {
      reflections,
      sessions,
      creator,
      lastMessage: reflections?.[0]?.content || null,
      lastTask: creator?.[0]?.taskType || null,
      blockState: sessions?.[0]?.block || null,
    };

  } catch (err) {
    console.error("Memory load failure:", err);
    return {
      reflections: [],
      sessions: [],
      creator: [],
      lastMessage: null,
      lastTask: null,
      blockState: null,
      error: String(err),
    };
  }
}


// ================================
// WRITE MEMORY ENTRY
// ================================
export async function saveMemoryEntry(userId, entry) {
  try {
    const { error } = await supabase
      .from("reflections")
      .insert([
        {
          userId,
          content: entry.output || entry.prompt,
          taskType: entry.taskType,
          engine: "v7",
          created_at: entry.ts,
        },
      ]);

    if (error) throw error;

    return { ok: true };
  } catch (err) {
    console.error("Memory write failed:", err);
    return { ok: false, error: String(err) };
  }
}