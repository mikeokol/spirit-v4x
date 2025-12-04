// routes/live.js
// Spirit v7 — Live Coaching Flow (Supabase-persistent)

import express from "express";
import {
  recordLiveEvent
} from "../engine/memory.js";

const router = express.Router();
const activeSessions = new Map();

// ---------------------------------------------------------
// START LIVE SESSION
// ---------------------------------------------------------
router.post("/start", async (req, res) => {
  try {
    const { userId } = req.body;
    if (!userId) {
      return res.status(400).json({ ok: false, error: "Missing userId." });
    }

    const sessionId = crypto.randomUUID();

    activeSessions.set(sessionId, {
      userId,
      state: {
        phase: "init",
        expecting: null,
        goal: null,
        block: null,
        step: 0,
        lastRPE: null
      }
    });

    await recordLiveEvent(userId, {
      sessionId,
      type: "session_started"
    });

    res.json({
      ok: true,
      sessionId,
      message: "Live session initiated. Elite Founder mode engaged."
    });
  } catch (err) {
    console.error("[LIVE /start ERROR]", err);
    res.status(500).json({ ok: false, error: "Failed to start session." });
  }
});

// ---------------------------------------------------------
// MESSAGE HANDLER
// ---------------------------------------------------------
router.post("/message", async (req, res) => {
  try {
    const { userId, sessionId, message } = req.body;

    if (!activeSessions.has(sessionId)) {
      return res.status(400).json({
        ok: false,
        error: "No active live session for this user."
      });
    }

    const session = activeSessions.get(sessionId);
    const result = await handleLiveMessage({
      userId,
      session,
      message
    });

    await recordLiveEvent(userId, {
      sessionId,
      type: "user_message",
      message,
      state: session.state
    });

    res.json({
      ok: true,
      sessionId,
      reply: result.reply,
      state: session.state
    });
  } catch (err) {
    console.error("[LIVE /message ERROR]", err);
    res.status(500).json({ ok: false, error: "Live message processing failed." });
  }
});

// ---------------------------------------------------------
// LIVE FLOW CORE LOGIC
// ---------------------------------------------------------
async function handleLiveMessage({ userId, session, message }) {
  const state = session.state;
  const msg = message.toLowerCase().trim();

  // INIT → ask goal
  if (state.phase === "init") {
    state.phase = "goal_query";
    state.expecting = "goal";
    return {
      reply: "Quick check-in: what's your main focus — fat loss, muscle gain, or performance?"
    };
  }

  // GOAL
  if (state.phase === "goal_query") {
    const allowed = ["fat loss", "muscle gain", "performance"];

    if (!allowed.includes(msg)) {
      return { reply: "Say: fat loss, muscle gain, or performance." };
    }

    state.goal = msg;
    state.phase = "pre_session";
    state.expecting = "ready";

    return { reply: "Got it. When you're ready to begin, say “ready”." };
  }

  // READY → WARMUP
  if (state.phase === "pre_session" && msg === "ready") {
    state.phase = "warmup";
    state.block = "warmup";
    state.expecting = "done";
    return { reply: "Warmup: 3 minutes light movement. Say “done” when finished." };
  }

  // WARMUP DONE
  if (state.phase === "warmup" && msg === "done") {
    state.expecting = "rpe";
    return { reply: "Nice. Rate that warmup 1–10." };
  }

  // WARMUP RPE → BLOCK 1
  if (state.phase === "warmup" && state.expecting === "rpe") {
    state.lastRPE = Number(msg) || 0;
    state.phase = "main_block";
    state.block = "block1";
    state.expecting = "done";
    return { reply: "Block 1: 3 rounds — 30 sec squats + 30 sec brisk walk. Say “done” after all 3 rounds." };
  }

  // BLOCK 1 COMPLETE
  if (state.phase === "main_block" && state.block === "block1" && msg === "done") {
    state.expecting = "rpe";
    return { reply: "Good work. Rate Block 1 — 1–10." };
  }

  // BLOCK 1 RPE → BLOCK 2
  if (state.phase === "main_block" && state.block === "block1" && state.expecting === "rpe") {
    state.lastRPE = Number(msg) || 0;
    state.block = "block2";
    state.expecting = "done";
    return { reply: "Block 2: 3 rounds — 20 sec push-ups + 40 sec brisk walk. Say “done” after all 3 rounds." };
  }

  // BLOCK 2 COMPLETE
  if (state.phase === "main_block" && state.block === "block2" && msg === "done") {
    state.expecting = "rpe";
    return { reply: "Strong. Rate Block 2 — 1–10." };
  }

  // BLOCK 2 RPE → BLOCK 3
  if (state.phase === "main_block" && state.block === "block2" && state.expecting === "rpe") {
    state.lastRPE = Number(msg) || 0;
    state.block = "block3";
    state.expecting = "done";
    return { reply: "Block 3: 2 rounds — 45 sec jog + 15 sec sprint. Say “done” when finished." };
  }

  // BLOCK 3 COMPLETE → COOLDOWN
  if (state.phase === "main_block" && state.block === "block3" && msg === "done") {
    state.phase = "cooldown";
    state.expecting = "done";
    return { reply: "Cooldown: 2 minutes slow breathing + stretching. Say “done” when finished." };
  }

  // COOLDOWN → COMPLETE
  if (state.phase === "cooldown" && msg === "done") {
    state.phase = "complete";
    return { reply: "Session complete. Amazing work — consistency is becoming your identity." };
  }

  return { reply: "I'm with you — continue." };
}

export default router;