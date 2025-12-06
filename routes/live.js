// routes/live.js
// Spirit v7 — Live Coaching Flow (Hybrid Elite Mode + Supabase persistence)

import express from "express";
import {
  recordLiveEvent,
  getUserLiveHistory,
  clearLiveHistory,
} from "../engine/memory.js";
import {
  analyzeLiveMessage,
  generateLiveCoachResponse,
} from "../engine/liveCoachingBrain.js";

const router = express.Router();

// In-memory session states (NOT logs)
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
        lastRPE: null,
        set: 0,
        setsInBlock: 0,
      },
    });

    await recordLiveEvent(userId, {
      sessionId,
      type: "session_started",
    });

    res.json({
      ok: true,
      sessionId,
      message: "Live session initiated. Elite Founder mode engaged.",
    });
  } catch (err) {
    console.error("[LIVE /start ERROR]", err);
    res
      .status(500)
      .json({ ok: false, error: "Failed to start session." });
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
        error: "No active live session for this user.",
      });
    }

    const session = activeSessions.get(sessionId);

    const result = await handleLiveMessage({
      userId,
      session,
      message,
    });

    await recordLiveEvent(userId, {
      sessionId,
      type: "user_message",
      message,
      state: session.state,
    });

    res.json({
      ok: true,
      sessionId,
      reply: result.reply,
      state: session.state,
    });
  } catch (err) {
    console.error("[LIVE /message ERROR]", err);
    res.status(500).json({
      ok: false,
      error: "Live message processing failed.",
    });
  }
});

// ---------------------------------------------------------
// END SESSION
// ---------------------------------------------------------
router.post("/end", async (req, res) => {
  try {
    const { userId, sessionId } = req.body;

    activeSessions.delete(sessionId);

    await recordLiveEvent(userId, {
      sessionId,
      type: "session_ended",
    });

    res.json({ ok: true, message: "Live session ended. Spirit rests." });
  } catch (err) {
    console.error("[LIVE /end ERROR]", err);
    res.status(500).json({ ok: false, error: "Failed to end live session." });
  }
});

// ---------------------------------------------------------
// LIVE FLOW CORE LOGIC (Hybrid: state machine + AI brain)
// ---------------------------------------------------------
async function handleLiveMessage({ userId, session, message }) {
  const state = session.state;
  const raw = (message || "").toString();
  const msg = raw.toLowerCase().trim();

  // -----------------------------
  // 1. PURE STATE MACHINE PATHS
  // -----------------------------

  // INIT → ask goal
  if (state.phase === "init") {
    state.phase = "goal_query";
    state.expecting = "goal";
    return {
      reply:
        "Quick check-in: what's your main focus — fat loss, muscle gain, or performance?",
    };
  }

  // GOAL SELECTION
  if (state.phase === "goal_query") {
    const allowed = ["fat loss", "muscle gain", "performance"];
    if (!allowed.includes(msg)) {
      // We'll later let AI answer, but here we give a strict nudge first
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
    state.set = 0;
    state.setsInBlock = 1; // single warmup block

    return {
      reply:
        "Warmup: 3 minutes light movement. Say “done” when finished.",
    };
  }

  // WARMUP DONE
  if (state.phase === "warmup" && msg === "done") {
    state.expecting = "rpe";
    return { reply: "Nice. Rate that warmup 1–10." };
  }

  // WARMUP RPE → BLOCK 1 (set-by-set)
  if (state.phase === "warmup" && state.expecting === "rpe") {
    state.lastRPE = Number(msg) || 0;
    state.phase = "main_block";
    state.block = "block1";
    state.expecting = "done";
    state.set = 1;
    state.setsInBlock = 3;

    return {
      reply:
        "Block 1 — Set 1/3: 30 sec squats + 30 sec brisk walk. Say “done” when you finish this set.",
    };
  }

  // BLOCK 1 — set-by-set
  if (state.phase === "main_block" && state.block === "block1") {
    if (msg === "done") {
      if (state.set < state.setsInBlock) {
        state.set += 1;
        state.expecting = "done";
        return {
          reply: `Strong. Set ${state.set - 1}/${state.setsInBlock} complete. When you're ready, start Set ${state.set}/${state.setsInBlock}: 30 sec squats + 30 sec brisk walk. Say “done” when finished.`,
        };
      } else {
        // finished all sets
        state.expecting = "rpe";
        return { reply: "Good work. Rate Block 1 — 1–10." };
      }
    }

    if (state.expecting === "rpe") {
      state.lastRPE = Number(msg) || 0;
      state.block = "block2";
      state.expecting = "done";
      state.set = 1;
      state.setsInBlock = 3;

      return {
        reply:
          "Block 2 — Set 1/3: 20 sec push-ups + 40 sec brisk walk. Say “done” when you finish this set.",
      };
    }
  }

  // BLOCK 2 — set-by-set
  if (state.phase === "main_block" && state.block === "block2") {
    if (msg === "done") {
      if (state.set < state.setsInBlock) {
        state.set += 1;
        state.expecting = "done";
        return {
          reply: `Nice. Set ${state.set - 1}/${state.setsInBlock} of Block 2 done. When ready, start Set ${state.set}/${state.setsInBlock}: 20 sec push-ups + 40 sec brisk walk. Say “done” when finished.`,
        };
      } else {
        state.expecting = "rpe";
        return { reply: "Strong. Rate Block 2 — 1–10." };
      }
    }

    if (state.expecting === "rpe") {
      state.lastRPE = Number(msg) || 0;
      state.block = "block3";
      state.expecting = "done";
      state.set = 1;
      state.setsInBlock = 2;

      return {
        reply:
          "Block 3 — Set 1/2: 45 sec fast walk or jog + 15 sec high-knee sprint. Say “done” when finished.",
      };
    }
  }

  // BLOCK 3 — set-by-set
  if (state.phase === "main_block" && state.block === "block3") {
    if (msg === "done") {
      if (state.set < state.setsInBlock) {
        state.set += 1;
        state.expecting = "done";
        return {
          reply: `You’re pushing well. Set ${state.set - 1}/${state.setsInBlock} done. When ready, start Set ${state.set}/${state.setsInBlock}: 45 sec fast walk or jog + 15 sec high-knee sprint. Say “done” when finished.`,
        };
      } else {
        // finished all sets → cooldown
        state.phase = "cooldown";
        state.expecting = "done";
        return {
          reply:
            "Cooldown: 2 minutes slow breathing + light stretching. Say “done” when finished.",
        };
      }
    }
  }

  // COOLDOWN DONE → COMPLETE
  if (state.phase === "cooldown" && msg === "done") {
    state.phase = "complete";
    state.expecting = "done";
    return {
      reply:
        "Session complete. Amazing work — consistency is becoming your identity.",
    };
  }

  // -----------------------------------------
  // 2. AI COACHING BRAIN FALLBACK (HYBRID)
  // -----------------------------------------
  // Anything that didn't match a strict flow goes here.
  try {
    const type = await analyzeLiveMessage({ message: raw, state });

    // If user said something like "next", "continue", etc. → map to action
    if (type === "action") {
      if (state.expecting === "done") {
        // Treat as "done" in current context
        return await handleLiveMessage({
          userId,
          session,
          message: "done",
        });
      }
      if (state.expecting === "ready") {
        return await handleLiveMessage({
          userId,
          session,
          message: "ready",
        });
      }
      // If it's action but we don't know where → just motivate
    }

    // Otherwise generate context-aware coaching answer
    const coachReply = await generateLiveCoachResponse({
      type,
      message: raw,
      state,
    });

    return { reply: coachReply };
  } catch (err) {
    console.error("[LIVE coaching-fallback ERROR]", err);
    // Absolute last resort — keep session alive
    return { reply: "I'm with you — continue. When you're ready, say “done” or “ready” to move on." };
  }
}

export default router;