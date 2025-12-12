// engine/executor.js
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
import { classifyIntent, scoreConfidence, buildECM } from "./engine/intentEngine.js"; // ← new

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
    const toneMap = {
      venting: "calm, validating, no advice unless asked",
      decision: "clear, concise, pros/cons",
      planning: "step-by-step, time-boxed",
      reflection: "open, gentle, mirroring",
      exploratory: "curious, Socratic",
      general: "neutral, helpful",
    };
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

    return { ok: true, reply, meta: { intent, confidence, ecm } }; // ← invisible polish
  } catch (err) {
    console.error("❌ Spirit Executor Error:", err);
    return { ok: false, error: "Executor failure." };
}
