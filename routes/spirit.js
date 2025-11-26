import express from "express";

// Engine modules
import { spiritController } from "../engine/controller.js";
import { spiritPlanner } from "../engine/planner.js";
import { spiritExecutor } from "../engine/executor.js";
import { spiritCritic } from "../engine/critic.js";
import { pullUserMemory, saveMemoryEntry } from "../engine/memory.js";

// Tools
import { workoutBuilder } from "../engine/tools/workoutBuilder.js";
import { creatorTools } from "../engine/tools/creatorTools.js";
import { reflectionTools } from "../engine/tools/reflectionTools.js";
import { analyticsTools } from "../engine/tools/analyticsTools.js";

const router = express.Router();

router.post("/", async (req, res) => {
  try {
    const { prompt, userId, taskType, context = {}, tone = "default" } = req.body;

    if (!prompt || !userId) {
      return res.status(400).json({
        ok: false,
        error: "Missing required fields: prompt, userId"
      });
    }

    // 1) CONTROLLER — interpret prompt → task
    const resolvedTask = taskType || spiritController(prompt);

    // 2) MEMORY — get last reflections/workouts/creator logs
    const memory = await pullUserMemory(userId);

    // 3) PLANNER — build reasoning plan
    const plan = await spiritPlanner(
      resolvedTask,
      prompt,
      memory,
      context
    );

    // 4) EXECUTOR — run task using toolbox
    const output = await spiritExecutor(plan, {
      prompt,
      userId,
      tone,
      context,
      memory,
      tools: {
        workoutBuilder,
        creatorTools,
        reflectionTools,
        analyticsTools
      }
    });

    // 5) CRITIC — validate final answer
    const review = await spiritCritic(output, plan, memory);

    // Optional memory write-back
    await saveMemoryEntry(userId, {
      prompt,
      output,
      taskType: resolvedTask,
      ts: new Date().toISOString()
    });

    // If critic says it's ok
    if (review.ok) {
      return res.json({
        ok: true,
        output,
        plan,
        review
      });
    }

    // If critic not satisfied
    return res.json({
      ok: true,
      output,
      plan,
      review,
      status: "revision"
    });

  } catch (err) {
    console.error("Spirit v7 Engine Error:", err);
    return res.status(500).json({
      ok: false,
      error: "Spirit v7 engine failed.",
      details: String(err)
    });
  }
});

export default router;