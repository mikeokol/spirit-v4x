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
import { classifyIntent, scoreConfidence, buildECM } from "./intentEngine.js"; // ← FIXED PATH
import { parseFitnessResponse } from "./fitnessParser.js";                   // ← NEW IMPORT

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// -----------------------------------------------------
// MAIN EXECUTOR
// -----------------------------------------------------
export async function runSpirit({ userId, message, mode, taskType, meta = {} }) {
  try {
    const userMemory = await loadMemory(userId);

    // --- high-leverage polish -------------------------------------------
    const intent = classifyIntent(message);
    const confidence = scoreConfidence(userId, message, meta);
    const ecm = buildECM(userId, intent, meta.stopReason || "user_sent");
    const systemPrompt = buildSystemPrompt({ mode, taskType, userMemory, intent, confidence, ecm });
    // ---------------------------------------------------------------------

    const completion = await client.chat.completions.create({
      model: "gpt-4.1",
      temperature: 0.4,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: message },
      ],
    });

    const reply = completion?.choices?.[0]?.message?.content || "…";

    // Persist memory + ECM anchor
    await handleMemoryUpdates({
      userId,
      mode,
      taskType,
      reply,
      userMemory,
      ecm, // ← resume anchor
    });

    // -----------------------------------------------------------------
    // BUILD RESPONSE OBJECT
    // -----------------------------------------------------------------
    let responseData = { ok: true, reply, meta: { intent, confidence, ecm } };

    // Parse fitness data if it's a workout plan
    if (mode === "fitness" && taskType === "workout_plan") {
      const parsedFitness = parseFitnessResponse(reply);
      if (parsedFitness) {
        responseData.plan = parsedFitness;
      }
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
  // existing mode saves
  if (mode === "sanctuary") await saveMemory(userId, { ...userMemory, lastSanctuary: reply });
  if (mode === "reflection") await saveReflectionHistory(userId, reply);
  if (mode === "creator") await saveCreatorHistory(userId, { input: "N/A", output: reply, taskType });
  if (mode === "hybrid") await saveHybridHistory(userId, { output: reply, taskType });
  if (mode === "fitness" && taskType === "workout_plan") await saveFitnessPlan(userId, reply);

  // Explicit Continuity Marker (invisible)
  if (ecm) await saveMemory(userId, { ...userMemory, lastECM: ecm });
}

// -----------------------------------------------------
// SYSTEM PROMPTS  (updated with intent + confidence + ECM)
// -----------------------------------------------------
function buildSystemPrompt({ mode, taskType, userMemory, intent, confidence, ecm }) {
  const taskPrompts = {
    sanctuary: "Soft, calm, grounding. You mirror emotion gently and create safety.",
    reflection: "Ask probing questions, explore patterns, help user gain clarity.",
    hybrid: "Blend fitness, mindset, planning, and creative structure into one plan.",
    fitness: "Design safe, structured workouts. Include warmup, sets, reps, rest, cooldown.",
    creator: "Write hooks, scripts, content strategy, thumbnails, and retention advice.",
    live: "NOT USED — voice/live handled in live.js.",
  };

  let base = `
You are SPIRIT v7.2 — an elite, adaptive, emotionally intelligent Life OS.
Stay precise, grounded, safe, and user-aware at all times.
`;

  if (userMemory) {
    base += `\nUser Context:\n${JSON.stringify(userMemory, null, 2)}\n`;
  }

  // Intent-boundary awareness (invisible polish)
  base += `\nIntent: ${intent.type} (confidence ${confidence}). Tone: ${
    {venting:"calm,validating",decision:"clear,concise",planning:"step-by-step",reflection:"open,gentle",exploratory:"curious,Socratic",general:"neutral,helpful"}[intent.type]
  }.\n`;

  // Explicit Continuity Marker (invisible, but guides re-frame)
  if (ecm) {
    base += `\nResume Anchor: ${ecm.resumeCue}\n`;
  }

  return `
${base}

Mode: ${mode}
Task: ${taskType}

${taskPrompts[mode] || ""}
`;
}
