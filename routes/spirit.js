// routes/spirit.js — Spirit v7 Unified Cognitive Engine Route

import express from "express";

// Engine
import { spiritController } from "../engine/controller.js";
import { spiritPlanner } from "../engine/planner.js";
import { spiritExecutor } from "../engine/executor.js";
import { spiritCritic } from "../engine/critic.js";
import { pullUserMemory } from "../engine/memory.js";

// Tools
import { workoutBuilder } from "../engine/tools/workoutBuilder.js";
import { creatorTools } from "../engine/tools/creatorTools.js";
import { reflectionTools } from "../engine/tools/reflectionTools.js";
import { analyticsTools } from "../engine/tools/analyticsTools.js";

const router = express.Router();

// SIMPLE CHAT FALLBACK — JS VERSION
async function simpleSpiritChat(message, memory) {
  return `
I hear you. Here's a quick response without entering deep analysis:

${message}

If you'd like deeper reasoning or guidance, ask me something more specific or emotional.
  `.trim();
}

router.post("/", async (req, res) => {
  try {
    const { message, mode, userId, context, tone, taskType: explicitTaskType } = req.body || {};

    if (!message && !mode && !explicitTaskType) {
      return res.status(400).json({ error: "No message, mode, or taskType provided." });
    }

    // 1. CONTROLLER → task type
    const taskType = explicitTaskType || spiritController(message, mode);

    // 2. MEMORY
    const memory = await pullUserMemory(userId);

    // 3. SIMPLE CHAT SHORT-CIRCUIT
    if (taskType === "simple_chat") {
      const reply = await simpleSpiritChat(message, memory);
      return res.json({ ok: true, reply, taskType, mode: mode || "sanctuary" });
    }

    // 4. PLANNER
    const plan = await spiritPlanner(taskType, {
      message,
      mode,
      tone,
      memory,
      userId,
      context,
    });

    // 5. EXECUTOR
    const draft = await spiritExecutor(plan, memory, {
      workoutBuilder,
      creatorTools,
      reflectionTools,
      analyticsTools,
    });

    // 6. CRITIC
    const review = await spiritCritic(draft, {
      userId,
      mode,
      taskType,
    });

    // 7. FINAL RESPONSE
    return res.json({
      ok: true,
      reply: draft,
      plan,
      review,
      taskType,
      mode: mode || "sanctuary",
    });
  } catch (err) {
    console.error("Spirit v7 Engine Error:", err);
    return res.status(500).json({
      ok: false,
      error: "Spirit engine failed.",
      details: String(err?.message || err),
    });
  }
});

export default router;