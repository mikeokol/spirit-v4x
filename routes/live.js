// routes/live.js — Live Elite Founder coaching sessions
import express from "express";
import { v4 as uuidv4 } from "uuid";

import { runSpiritEngine } from "../engine/controller.js";
import { recordLiveEvent } from "../engine/memory.js";

const router = express.Router();

// In-memory map of active live coaching sessions.
// Key: sessionId -> { userId, startedAt, lastMessageAt }
const activeSessions = new Map();

/**
 * POST /live/start
 * Body: { userId: string }
 */
router.post("/start", async (req, res) => {
  try {
    const { userId } = req.body || {};
    if (!userId) {
      return res.status(400).json({
        ok: false,
        error: "Missing userId in request body.",
      });
    }

    const sessionId = uuidv4();
    const now = new Date().toISOString();

    activeSessions.set(sessionId, {
      userId,
      startedAt: now,
      lastMessageAt: now,
    });

    // Persist event to memory (best-effort; non-blocking for UX)
    recordLiveEvent(userId, {
      type: "session_started",
      sessionId,
    }).catch((err) => {
      console.warn("[live] recordLiveEvent(session_started) failed:", err);
    });

    return res.json({
      ok: true,
      sessionId,
      message: "Live session initiated. Elite Founder mode engaged.",
    });
  } catch (err) {
    console.error("[live] /start error:", err);
    return res.status(500).json({
      ok: false,
      error: "Failed to start live session.",
    });
  }
});

/**
 * POST /live/message
 * Body: { userId: string, sessionId: string, message: string }
 */
router.post("/message", async (req, res) => {
  try {
    const { userId, sessionId, message } = req.body || {};

    if (!userId || !sessionId || !message) {
      return res.status(400).json({
        ok: false,
        error: "Missing userId, sessionId, or message in request body.",
      });
    }

    const session = activeSessions.get(sessionId);
    if (!session || session.userId !== userId) {
      return res.status(400).json({
        ok: false,
        error: "No active live session for this user.",
      });
    }

    session.lastMessageAt = new Date().toISOString();

    const result = await runSpiritEngine({
      userId,
      message,
      mode: "live",
      taskType: "live_coaching",
    });

    const reply =
      typeof result?.reply === "string"
        ? result.reply
        : "Session active, but no reply generated.";

    // Persist coach turn
    recordLiveEvent(userId, {
      type: "coach_turn",
      sessionId,
      reply,
    }).catch((err) => {
      console.warn("[live] recordLiveEvent(coach_turn) failed:", err);
    });

    return res.json({
      ok: true,
      sessionId,
      reply,
    });
  } catch (err) {
    console.error("[live] /message error:", err);
    return res.status(500).json({
      ok: false,
      error: "Live coaching message failed.",
    });
  }
});

/**
 * POST /live/end
 * Body: { userId: string, sessionId: string }
 */
router.post("/end", async (req, res) => {
  try {
    const { userId, sessionId } = req.body || {};

    if (!userId || !sessionId) {
      return res.status(400).json({
        ok: false,
        error: "Missing userId or sessionId in request body.",
      });
    }

    const session = activeSessions.get(sessionId);
    if (!session || session.userId !== userId) {
      return res.status(400).json({
        ok: false,
        error: "No active live session.",
      });
    }

    activeSessions.delete(sessionId);

    recordLiveEvent(userId, {
      type: "session_ended",
      sessionId,
    }).catch((err) => {
      console.warn("[live] recordLiveEvent(session_ended) failed:", err);
    });

    return res.json({
      ok: true,
      message: "Session ended. Spirit will rest until called again.",
    });
  } catch (err) {
    console.error("[live] /end error:", err);
    return res.status(500).json({
      ok: false,
      error: "Failed to end live session.",
    });
  }
});

export default router;