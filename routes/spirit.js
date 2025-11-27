import express from "express";

// Engine modules
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

// SIMPLE FALLBACK CHAT
async function simpleSpiritChat(message) {
  return `I hear you. What direction do you want to explore?`;
}

router.post("/", async (req, res) => {
  try {
    const { prompt, message, mode, userId, taskType, tone, context } = req.body;

    // Handle aliasing (prompt vs message)
    const input = message ?? prompt;
    if (!input) {
      return res.status(400).json({ error: "No message provided." });
    }

    // CONTROLLER
    const resolvedTask = taskType ?? spiritController(input, mode);

    // MEMORY
    const memory = await pullUserMemory(userId);

    // SIMPLE CHAT
    if (resolvedTask === "simple_chat") {
      const reply = await simpleSpiritChat(input);
      return res.json({ reply });
    }

    // PLANNER
    const plan = await spiritPlanner(resolvedTask, {
      message: input,
      mode,
      tone,
      memory,
      userId,
      context,
    });

    // EXECUTOR — MUST PASS TOOLS
    const draft = await spiritExecutor(plan, memory, {
      workoutBuilder,
      creatorTools,
      reflectionTools,
      analyticsTools,
    });

    // CRITIC
    const review = await spiritCritic(draft, memory);

    if (!review.ok) {
      return res.json({
        reply: draft,
        notes: review.notes,
        status: "revision_requested",
      });
    }

    return res.json({
      reply: draft,
      plan,
      review,
    });

  } catch (err) {
    console.error("Spirit v7 Engine Error:", err);
    return res.status(500).json({
      error: "Spirit engine failed.",
      details: String(err),
    });
  }
});

export default router;