// engine/executor.js
// Spirit v7 — Core Cognitive Executor (production clean)

import OpenAI from "openai";
import {
  loadMemory,
  saveMemory,
  saveCreatorHistory,
  saveReflectionHistory,
  saveHybridHistory,
  saveFitnessPlan,
} from "./memory.js";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// -----------------------------------------------------
// MAIN EXECUTOR
// -----------------------------------------------------
export async function runSpirit({ userId, message, mode, taskType }) {
  try {
    const userMemory = await loadMemory(userId);

    const systemPrompt = buildSystemPrompt({
      mode,
      taskType,
      userMemory,
    });

    const completion = await client.chat.completions.create({
      model: "gpt-4.1",
      temperature: 0.4,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: message },
      ],
    });

    const reply =
      completion?.choices?.[0]?.message?.content || "…";

    // Persist memory based on mode & task
    await handleMemoryUpdates({
      userId,
      mode,
      taskType,
      reply,
      userMemory,
    });

    return { ok: true, reply };
  } catch (err) {
    console.error("❌ Spirit Executor Error:", err);
    return { ok: false, error: "Executor failure." };
  }
}

// -----------------------------------------------------
// MEMORY PIPELINE
// -----------------------------------------------------
async function handleMemoryUpdates({
  userId,
  mode,
  taskType,
  reply,
  userMemory,
}) {
  // Sanctuary / Reflection → save logs
  if (mode === "sanctuary") {
    await saveMemory(userId, {
      ...userMemory,
      lastSanctuary: reply,
    });
  }

  if (mode === "reflection") {
    await saveReflectionHistory(userId, reply);
  }

  // Creator → history
  if (mode === "creator") {
    await saveCreatorHistory(userId, {
      input: "N/A",
      output: reply,
      taskType,
    });
  }

  // Hybrid → history
  if (mode === "hybrid") {
    await saveHybridHistory(userId, {
      output: reply,
      taskType,
    });
  }

  // Fitness → last plan
  if (mode === "fitness" && taskType === "workout_plan") {
    await saveFitnessPlan(userId, reply);
  }
}

// -----------------------------------------------------
// SYSTEM PROMPTS
// -----------------------------------------------------
function buildSystemPrompt({ mode, taskType, userMemory }) {
  let base = `
You are SPIRIT v7.1 — an elite, adaptive, emotionally intelligent Life OS.
Stay precise, grounded, safe, and user-aware at all times.
`;

  if (userMemory) {
    base += `\nUser Context:\n${JSON.stringify(userMemory, null, 2)}\n`;
  }

  const modePrompts = {
    sanctuary:
      "Soft, calm, grounding. You mirror emotion gently and create safety.",
    reflection:
      "Ask probing questions, explore patterns, help user gain clarity.",
    hybrid:
      "Blend fitness, mindset, planning, and creative structure into one plan.",
    fitness:
      "Design safe, structured workouts. Include warmup, sets, reps, rest, cooldown.",
    creator:
      "Write hooks, scripts, content strategy, thumbnails, and retention advice.",
    live:
      "NOT USED — voice/live handled in live.js.",
  };

  const taskPrompts = {
    sanctuary_chat: "Reply with 1–2 paragraphs max. No lists unless needed.",
    reflection_chat:
      "Ask thoughtful questions. Invite introspection and emotional clarity.",
    creator_script:
      "Provide hooks, scripts, thumbnails, angles, delivery notes.",
    workout_plan:
      "Return a complete workout with sets, reps, rest, warmup, cooldown.",
    hybrid_plan:
      "Blend personal, creator, fitness, and mindset actions into a unified day plan.",
  };

  return `
${base}

Mode: ${mode}
Task: ${taskType}

${modePrompts[mode] || ""}
${taskPrompts[taskType] || ""}
`;
}