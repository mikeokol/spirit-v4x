// routes/debug.js
// Spirit v7 — Debug Tools (Safe, API-Stable)

import express from "express";
import {
  getUserLiveHistory,
  clearLiveHistory
} from "../engine/memory.js";

const router = express.Router();

// -----------------------------------------------------
// PING
// -----------------------------------------------------
router.get("/ping", (req, res) => {
  res.json({ ok: true, msg: "debug:pong" });
});

// -----------------------------------------------------
// LIVE SESSION HISTORY BY USER
// -----------------------------------------------------
router.get("/live/history", async (req, res) => {
  try {
    const userId = req.query.userId;
    if (!userId) {
      return res.status(400).json({
        ok: false,
        error: "Missing ?userId="
      });
    }

    const history = await getUserLiveHistory(userId);

    res.json({
      ok: true,
      userId,
      history
    });
  } catch (err) {
    console.error("[DEBUG /live/history ERROR]", err);
    res.status(500).json({
      ok: false,
      error: "Failed to fetch live history."
    });
  }
});

// -----------------------------------------------------
// CLEAR LIVE SESSION HISTORY
// -----------------------------------------------------
router.post("/live/clear", async (req, res) => {
  try {
    const { userId } = req.body;

    if (!userId) {
      return res.status(400).json({
        ok: false,
        error: "Missing userId."
      });
    }

    await clearLiveHistory(userId);

    res.json({ ok: true, message: "Live session history cleared." });
  } catch (err) {
    console.error("[DEBUG /live/clear ERROR]", err);
    res.status(500).json({
      ok: false,
      error: "Failed to clear live history."
    });
  }
});

export default router;