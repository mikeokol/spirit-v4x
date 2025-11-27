// routes/spirit.js — Unified Spirit v7.1 HTTP entrypoint

import { Router } from "express";
import { runSpiritEngine } from "../engine/controller.js";

const router = Router();

router.post("/", async (req, res) => {
  try {
    const { userId, message, mode, taskType } = req.body || {};

    if (!userId || !message) {
      return res.status(400).json({
        ok: false,
        error: "Missing userId or message in request body.",
      });
    }

    const result = await runSpiritEngine({
      userId,
      message,
      mode,
      taskType,
    });

    return res.json(result);
  } catch (err) {
    console.error("[/spirit] error:", err);
    return res.status(500).json({
      ok: false,
      error: "Spirit engine failed.",
      details: err?.message || String(err),
    });
  }
});

export default router;