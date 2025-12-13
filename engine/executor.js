// backend/engine/executor.js
// Spirit v7 — Core Cognitive Executor (production clean + polish)

import OpenAI from "openai";
import {
  loadMemory,
  saveMemory,
  saveCreatorHistory,
  saveReflectionHistory,
  saveHybridHistory,
  saveFitnessPlan,
} from "./memory.js";
import { classifyIntent, scoreConfidence, buildECM } from "./intentEngine.js";
import { parseFitnessResponse } from "./fitnessParser.js";   // ← NEW

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// -----------------------------------------------------
// MAIN EXECUTOR
// -----------------------------------------------------
export async function runSpirit({ userId, message, mode, taskType, meta = {} }) {
  try {
    const userMemory = await loadMemory(userId);

    const intent = classifyIntent(message);
    const confidence = scoreConfidence(userId, message, meta);
    const ecm = buildECM(userId, intent, meta.stopReason || "user_sent");
    const systemPrompt = buildSystemPrompt({ mode, taskType, userMemory, intent, confidence, ecm });

    const completion = await client.chat.completions.create({
      model: "gpt-4.1",
      temperature: 0.4,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: message },
      ],
    });

    const reply = completion?.choices?.[0]?.message?.content || "…";

    await handleMemoryUpdates({
      userId,
      mode,
      taskType,
      reply,
      userMemory,
      ecm,
    });

    // -----------------------------------------------------------------
    // BUILD RESPONSE OBJECT
    // -----------------------------------------------------------------
    let responseData = { ok: true, reply, meta: { intent, confidence, ecm } };

    // Parse fitness data if it's a workout plan
    if (mode === "fitness" && taskType === "workout_plan") {
      const parsed = parseFitnessResponse(reply);
      if (parsed) responseData.plan = parsed;
    }

    return responseData;
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
  ecm,
}) {
  if (mode === "sanctuary") await saveMemory(userId, { ...userMemory, lastSanctuary: reply });
  if (mode === "reflection") await saveReflectionHistory(userId, reply);
  if (mode === "creator") await saveCreatorHistory(userId, { input: "N/A", output: reply, taskType });
  if (mode === "hybrid") await saveHybridHistory(userId, { output: reply, taskType });
  if (mode === "fitness" && taskType === "workout_plan") await saveFitnessPlan(userId, reply);
  if (ecm) await saveMemory(userId, { ...userMemory, lastECM: ecm });
}

// -----------------------------------------------------
// SYSTEM PROMPTS
// -----------------------------------------------------
function buildSystemPrompt({ mode, taskType, userMemory, intent, confidence, ecm }) {
  const taskPrompts = {
    sanctuary_chat: "Reply with 1–2 paragraphs max. No lists unless needed.",
    reflection_chat: "Ask thoughtful questions. Invite introspection and emotional clarity.",
    creator_script: "Provide hooks, scripts, thumbnails, angles, delivery notes.",
    workout_plan: "Return a complete workout with sets, reps, rest, warmup, cooldown.",
    hybrid_plan: "Blend personal, creator, fitness, and mindset actions into a unified day plan.",
  };

  const toneMap = {
    venting: "calm, validating",
    decision: "clear, concise",
    planning: "step-by-step",
    reflection: "open, gentle",
    exploratory: "curious, Socratic",
    general: "neutral, helpful",
  };

  let base = `
You are SPIRIT v7.2 — an elite, adaptive, emotionally intelligent Life OS.
Stay precise, grounded, safe, and user-aware at all times.
`;

  if (userMemory) {
    base += `\nUser Context:\n${JSON.stringify(userMemory, null, 2)}\n`;
  }

  base += `\nIntent: ${intent.type} (confidence ${confidence}). Tone: ${toneMap[intent.type] || "neutral, helpful"}.\n`;

  if (ecm) {
    base += `\nResume Anchor: ${ecm.resumeCue}\n`;
  }

  return `
${base}

Mode: ${mode}
Task: ${taskType}

${taskPrompts[taskType] || ""}
`;
}
