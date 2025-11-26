// controller.js — Trinity v7 Cognitive Engine Controller
// Purpose: interpret user prompt → taskType for Planner

export function spiritController(prompt = "") {
  const text = prompt.toLowerCase().trim();

  // --- Explicit Command Patterns ---
  if (text.includes("workout") || text.includes("gym") || text.includes("training")) {
    return "fitness_task";
  }

  if (text.includes("script") || text.includes("content") || text.includes("creator")) {
    return "creator_task";
  }

  if (text.includes("reflect") || text.includes("journal") || text.includes("insight")) {
    return "reflection_task";
  }

  if (text.includes("live session") || text.includes("coach me") || text.includes("next set")) {
    return "live_session_task";
  }

  if (text.includes("analytics") || text.includes("stats") || text.includes("progress")) {
    return "analytics_task";
  }

  if (text.includes("hybrid") || text.includes("combine") || text.includes("mix")) {
    return "hybrid_task";
  }

  // --- Generic Conversational Input ---
  // Default for typical chat interactions (Sanctuary)
  if (prompt.length <= 250) {
    return "sanctuary_chat";
  }

  // --- Long-Form / Reasoning / Planning ---
  return "analysis_task";
}