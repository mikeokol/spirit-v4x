// routes/spirit.js  (addition only – no breaking changes)
import express from "express";
import { runSpirit } from "../engine/executor.js";
import {
  saveCreatorHistory,
  saveReflectionHistory,
  saveHybridHistory,
  saveFitnessPlan,
} from "../engine/memory.js";
import { liveController } from "../controllers/liveController.js"; // ← NEW

const router = express.Router();

/* -----------------------------------------------------
   LIVE-MODE VALVE  (Prime-Directive compliant)
   ----------------------------------------------------- */
router.post("/live", liveController);   // physically separate route – no leak

/* -----------------------------------------------------
   ORIGINAL UNIFIED ENDPOINT  (untouched)
   ----------------------------------------------------- */
router.post("/", async (req, res) => {
  try {
    const { userId, message, mode, taskType } = req.body;
    if (!userId || !message || !mode || !taskType) {
      return res.status(400).json({ ok: false, error: "Missing required fields." });
    }
    const result = await runSpirit({ userId, message, mode, taskType });
    if (!result.ok) return res.status(500).json({ ok: false, error: "Spirit executor failed." });

    if (mode === "creator") await saveCreatorHistory(userId, { input: message, output: result.reply, taskType });
    if (mode === "reflection") await saveReflectionHistory(userId, result.reply);
    if (mode === "hybrid") await saveHybridHistory(userId, { input: message, output: result.reply, ts: Date.now() });
    if (mode === "fitness" && taskType === "workout_plan") await saveFitnessPlan(userId, result.reply);

    return res.json({ ok: true, ...result });
  } catch (err) {
    console.error("[Spirit Route Error]", err);
    return res.status(500).json({ ok: false, error: "Spirit route failed.", details: String(err?.message || err) });
  }
});

export default router;
