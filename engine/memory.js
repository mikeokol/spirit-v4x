// engine/memory.js — Spirit v7 Memory Loader (stubbed)

export async function pullUserMemory(userId) {
  // For now, return an empty but well-structured memory object.
  // You can later plug this into Supabase / Neon / Postgres as needed.
  return {
    reflections: [],
    sessions: [],
    userId: userId ?? null,
  };
}