// controllers/liveController.js — Spirit v5.1 Hybrid Live Coaching

import { spiritVoice } from "../utils/spiritVoice.js";

export const liveController = async (req, res) => {
  try {
    const {
      goal,          // e.g. "hypertrophy", "fat loss", "discipline"
      focus,         // e.g. "upper body", "morning routine"
      mood,          // e.g. "tired", "anxious", "focused"
      context,       // free text, optional
      duration       // in minutes, optional
    } = req.body;

    // Minimal required fields for a decent session
    if (!goal && !focus && !context) {
      return res.status(400).json({
        ok: false,
        error: "Provide at least one of: goal, focus, or context for the live session."
      });
    }

    const safeDuration = duration && duration > 0 ? duration : 10;

    // Simple hybrid structure: breath → identity → action → close
    const steps = [
      {
        type: "grounding",
        title: "Arrive",
        instruction: `Slow your breathing. Inhale for 4, hold for 2, exhale for 6. Do this for 5 cycles and let your body know you are safe.`,
      },
      {
        type: "identity",
        title: "Who You Are Becoming",
        instruction: `Name the identity you’re training into. For example: “I am the kind of person who shows up even when I feel ${mood || "resistance"}.” Say it out loud once.`,
      },
      {
        type: "clarify",
        title: "Clarify Today’s Focus",
        instruction: `In one sentence, define today’s target: “Today I will focus on ${focus || goal || "showing up fully"}.”`,
      },
      {
        type: "action",
        title: "Body Action Block",
        instruction: `Pick one concrete block that matches your goal. For example: 3 x 10 sets for a main lift (or 10–15 minutes of focused movement) that points toward ${goal || "your current direction"}. No perfection, just completion.`,
      },
      {
        type: "reflection",
        title: "Lock It In",
        instruction: `After you finish, answer this in one line: “What did I do today that Future Me will thank me for?” Capture it somewhere, even in your notes.`,
      }
    ];

    const narration = spiritVoice(
      "live",
      `Hybrid coaching session for goal: ${goal || "alignment"}, focus: ${focus || "showing up"}, mood: ${mood || "unknown"}.`
    );

    return res.json({
      ok: true,
      mode: "live",
      duration: safeDuration,
      steps,
      narration
    });
  } catch (err) {
    res.status(500).json({
      ok: false,
      error: err.message
    });
  }
};
