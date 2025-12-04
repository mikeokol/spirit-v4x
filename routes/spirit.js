// routes/spirit.js
// Spirit v7 — Unified Mode Router (production clean)

import express from "express";
import { runSpirit } from "../engine/executor.js";
import {
  saveCreatorHistory,
  saveReflectionHistory,
  saveHybridHistory,
  saveFitnessPlan,
} from "../engine/memory.js";

const router = express.Router();

// -----------------------------------------------------
// SPIRIT MAIN ENDPOINT
// -----------------------------------------------------
router.post("/", async (req, res) => {
  try {
    const { userId, message, mode, taskType } = req.body;

    // Validate required fields
    if (!userId || !message || !mode || !taskType) {
      return res.status(400).json({
        ok: false,
        error: "Missing required fields.",
      });
    }

    // Run core Spirit logic
    const result = await runSpirit({ userId, message, mode, taskType });

    if (!result.ok) {
      return res.status(500).json({
        ok: false,
        error: "Spirit executor failed.",
      });
    }

    // -------------------------------------------------
    // MODE-SPECIFIC MEMORY HANDLERS
    // -------------------------------------------------

    // CREATOR MODE
    if (mode === "creator") {
      await saveCreatorHistory(userId, {
        input: message,
        output: result.reply,
        taskType,
      });
    }

    // REFLECTION MODE
    if (mode === "reflection") {
      await saveReflectionHistory(userId, result.reply);
    }

    // HYBRID MODE
    if (mode === "hybrid") {
      await saveHybridHistory(userId, {
        input: message,
        output: result.reply,
        ts: Date.now(),
      });
    }

    // FITNESS MODE — plan storage
    if (mode === "fitness" && taskType === "workout_plan") {
      await saveFitnessPlan(userId, result.reply);
    }

    // -------------------------------------------------
    // RETURN SUCCESS
    // -------------------------------------------------
    return res.json({ ok: true, ...result });

  } catch (err) {
    console.error("[Spirit Route Error]", err);

    return res.status(500).json({
      ok: false,
      error: "Spirit route failed.",
      details: String(err?.message || err),
    });
  }
});

export default router;