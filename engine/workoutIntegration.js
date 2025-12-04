// engine/workoutIntegration.js
// ---------------------------------------------------------
// Spirit v7 — Fitness → Live Integration Layer
// Connects workout plans to Live Mode with parsed structure
// ---------------------------------------------------------

import { parseWorkoutPlan, flattenWorkoutForLive } from "./workoutParser.js";
import { loadMemory, saveMemory } from "./memory.js";

// ----------------------------------------------
// Save a fitness plan + its parsed structure
// ----------------------------------------------
export async function saveParsedWorkout(userId, planText) {
  const parsed = parseWorkoutPlan(planText);

  const memory = await loadMemory(userId);

  await saveMemory(userId, {
    ...memory,
    lastWorkoutRaw: planText,
    lastWorkoutParsed: parsed.ok ? parsed.sections : null,
    lastWorkoutSequence: parsed.ok ? flattenWorkoutForLive(parsed) : [],
  });

  return parsed;
}

// ----------------------------------------------
// Load last workout sequence to use during Live coaching
// ----------------------------------------------
export async function loadWorkoutSequence(userId) {
  const memory = await loadMemory(userId);

  return (
    memory.lastWorkoutSequence ||
    [] // fallback
  );
}

// ----------------------------------------------
// Load raw plan (optional debugging)
// ----------------------------------------------
export async function loadLastWorkoutPlan(userId) {
  const memory = await loadMemory(userId);
  return memory.lastWorkoutRaw || "";
}