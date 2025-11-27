// engine/controller.js — Unified Spirit v7.1 engine

import { loadUserMemory, saveUserMemory } from "./memory.js";
import { runPlanner } from "./planner.js";
import { runExecutor } from "./executor.js";
import { runCritic } from "./critic.js";

/**
 * Main orchestrator for Spirit v7.1
 */
export async function runSpiritEngine({ userId, message, mode, taskType }) {
  const safeMode = mode || "sanctuary";
  const safeTask = taskType || "sanctuary_chat";

  // 1) Load memory
  const memory = await loadUserMemory(userId);

  // 2) Planner
  const plan = await runPlanner({
    userId,
    message,
    mode: safeMode,
    taskType: safeTask,
    memory,
  });

  // 3) Executor
  const reply = await runExecutor({
    userId,
    message,
    plan,
    memory,
    mode: safeMode,
    taskType: safeTask,
  });

  // 4) Critic
  const review = await runCritic({
    reply,
    plan,
    mode: safeMode,
    taskType: safeTask,
  });

  // 5) Memory patch
  const patch = {
    lastModes: Array.from(
      new Set([...(memory.lastModes || []), safeMode])
    ).slice(-10),
    lastTaskType: safeTask,
    lastReplySummary:
      typeof reply === "string" ? reply.slice(0, 200) : null,
  };

  if (safeTask === "workout_plan") patch.lastWorkoutPlan = reply;
  if (safeTask === "creator_script") patch.lastCreatorScript = reply;
  if (safeTask === "reflection_prompt") patch.lastReflection = reply;
  if (safeTask === "hybrid_plan") patch.lastHybridPlan = reply;

  await saveUserMemory(userId, patch);

  return {
    ok: true,
    reply,
    plan,
    review,
    taskType: safeTask,
    mode: safeMode,
  };
}