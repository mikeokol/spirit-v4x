// routes/live.js — Spirit v7.2 Live Coaching Route
import express from "express";
import { v4 as uuidv4 } from "uuid";

import {
  loadUserMemory,
  saveUserMemory,
  recordLiveEvent
} from "../engine/memory.js";

import { runLiveCoachTurn } from "../engine/liveCoach.js";

const router = express.Router();

// In-memory active sessions
// sessionId -> { userId, startedAt, lastMessageAt }
const activeSessions = new Map();

/**
 * POST /live/start
 * Body: { userId }
 */
router.post("/start", async (req, res) => {
  try {
    const { userId } = req.body || {};
    if (!userId) {
      return res.status(400).json({
        ok: false,
        error: "Missing userId.",
      });
    }

    const sessionId = uuidv4();
    const now = new Date().toISOString();

    activeSessions.set(sessionId, {
      userId,
      startedAt: now,
      lastMessageAt: now,
    });

    // Memory update (non-blocking)
    recordLiveEvent(userId, {
      type: "session_started",
      sessionId,
    }).catch(console.warn);

    return res.json({
      ok: true,
      sessionId,
      message: "Live session initiated. Elite Founder mode engaged.",
    });
  } catch (err) {
    console.error("[live/start]", err);
    res.status(500).json({
      ok: false,
      error: "Failed to start live session.",
    });
  }
});

/**
 * POST /live/message
 * Body: { userId, sessionId, message }
 */
router.post("/message", async (req, res) => {
  try {
    const { userId, sessionId, message } = req.body || {};

    if (!userId || !sessionId || !message) {
      return res.status(400).json({
        ok: false,
        error: "Missing userId, sessionId, or message.",
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

    // Run live engine for this turn
    const result = await runLiveCoachTurn({
      userId,
      sessionId,
      message,
    });

    const reply = result.reply || "Session active, but no reply generated.";

    // Memory update (non-blocking)
    recordLiveEvent(userId, {
      type: "coach_turn",
      sessionId,
      reply,
    }).catch(console.warn);

    res.json({
      ok: true,
      sessionId,
      reply,
      state: result.state,
    });
  } catch (err) {
    console.error("[live/message]", err);
    res.status(500).json({
      ok: false,
      error: "Live coaching message failed.",
    });
  }
});

/**
 * POST /live/end
 */
router.post("/end", async (req, res) => {
  try {
    const { userId, sessionId } = req.body || {};

    if (!userId || !sessionId) {
      return res.status(400).json({
        ok: false,
        error: "Missing userId or sessionId.",
      });
    }

    const session = activeSessions.get(sessionId);
    if (!session || session.userId !== userId) {
      return res.status(400).json({
        ok: false,
        error: "No active session.",
      });
    }

    activeSessions.delete(sessionId);

    recordLiveEvent(userId, {
      type: "session_ended",
      sessionId,
    }).catch(console.warn);

    return res.json({
      ok: true,
      message: "Live session ended. Spirit rests.",
    });
  } catch (err) {
    console.error("[live/end]", err);
    res.status(500).json({
      ok: false,
      error: "Failed to end session.",
    });
  }
});

export default router;