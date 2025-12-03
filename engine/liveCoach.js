// engine/liveCoach.js — Spirit v7.1 Live Coaching Brain (stepwise, no essays)

import { loadUserMemory, saveUserMemory } from "./memory.js";

/**
 * runLiveCoachTurn
 * This is called on every /live/message.
 *
 * It:
 *  - Reads previous live state from memory
 *  - Advances the workout / coaching flow
 *  - Returns ONE short instruction + updated state
 */
export async function runLiveCoachTurn({ userId, message }) {
  const rawMessage = (message || "").trim();
  const lower = rawMessage.toLowerCase();

  const memory = await loadUserMemory(userId);
  const currentLive = memory.live || {};
  const prevState = currentLive.state || null;

  let nextState = prevState || null;
  let reply = "";

  // -------------------------------------------------------
  // 1. No prior state → start the live coaching flow
  // -------------------------------------------------------
  if (!prevState) {
    reply =
      "Quick check-in: what’s your main focus right now — fat loss, muscle gain, or performance? Reply with one.";
    nextState = {
      phase: "goal_query",
      expecting: "goal",
      goal: null,
      block: null,
      step: 0,
    };
  }

  // -------------------------------------------------------
  // 2. Expecting goal selection
  // -------------------------------------------------------
  else if (prevState.expecting === "goal") {
    let goal = "general";

    if (lower.includes("fat")) goal = "fat_loss";
    else if (
      lower.includes("muscle") ||
      lower.includes("size") ||
      lower.includes("bulk") ||
      lower.includes("big")
    )
      goal = "muscle_gain";
    else if (
      lower.includes("perform") ||
      lower.includes("sport") ||
      lower.includes("speed") ||
      lower.includes("jump")
    )
      goal = "performance";

    reply =
      `Got it. We’ll train for ${goal.replace("_", " ")}.\n` +
      `When you’re ready to start, type “ready” and we’ll begin with a short warmup.`;

    nextState = {
      phase: "pre_session",
      expecting: "confirm_ready",
      goal,
      block: null,
      step: 0,
    };
  }

  // -------------------------------------------------------
  // 3. Waiting for READY
  // -------------------------------------------------------
  else if (prevState.expecting === "confirm_ready") {
    if (lower.includes("ready")) {
      reply =
        "Warmup: 3 minutes of light movement — walk, cycle, or jog at an easy pace. When you’re done, type “done”.";
      nextState = {
        ...prevState,
        phase: "warmup",
        block: "warmup",
        step: 1,
        expecting: "done",
      };
    } else {
      reply = 'When you’re ready to begin, just type “ready”.';
      nextState = prevState;
    }
  }

  // -------------------------------------------------------
  // 4. Waiting for DONE on warmup / blocks
  // -------------------------------------------------------
  else if (prevState.expecting === "done") {
    if (lower.includes("done")) {
      // After each block, ask RPE
      reply =
        "Nice. On a scale of 1–10, how hard did that last block feel? (1 = very easy, 10 = all-out).";
      nextState = {
        ...prevState,
        expecting: "rpe",
      };
    } else {
      reply =
        "Stay with this block. When you finish it, type “done” so we can move on.";
      nextState = prevState;
    }
  }

  // -------------------------------------------------------
  // 5. RPE input → adjust and move into main block
  // -------------------------------------------------------
  else if (prevState.expecting === "rpe") {
    const num = parseInt(lower, 10);
    let intensityNote = "Good baseline.";

    if (!isNaN(num)) {
      if (num <= 4) {
        intensityNote =
          "Very easy — we can safely push a bit harder in the next block.";
      } else if (num >= 8) {
        intensityNote =
          "That was intense — we’ll keep things controlled and smart on the next block.";
      } else {
        intensityNote = "Solid effort — we’ll hold a steady intensity.";
      }
    }

    let nextBlockText = "";
    const goal = prevState.goal || memory.preferences?.fitnessGoal || "general";

    if (goal === "fat_loss") {
      nextBlockText =
        `${intensityNote}\n` +
        "Block 1: 3 rounds — 30 seconds bodyweight squats, 30 seconds brisk walk in place. When all 3 rounds are done, type “done”.";
    } else if (goal === "muscle_gain") {
      nextBlockText =
        `${intensityNote}\n` +
        "Block 1: 3 sets — 10–12 slow, controlled squats and 10–12 push-ups (incline or on knees if needed). Rest ~60 seconds between sets. When all 3 sets are done, type “done”.";
    } else if (goal === "performance") {
      nextBlockText =
        `${intensityNote}\n` +
        "Block 1: 4 × 20m fast-but-controlled runs or jumps, walk back for recovery. When you’ve completed all 4, type “done”.";
    } else {
      nextBlockText =
        `${intensityNote}\n` +
        "Block 1: mix of squats, push-ups, and light cardio. Aim for 3 rounds. When you’ve completed them, type “done”.";
    }

    reply = nextBlockText;
    nextState = {
      phase: "main_block",
      goal,
      block: "block1",
      step: 1,
      expecting: "done",
    };
  }

  // -------------------------------------------------------
  // 6. Generic continuation logic for later blocks
  // -------------------------------------------------------
  else {
    if (lower.includes("done")) {
      reply =
        "Strong work. We can either stack another block or wrap here. Type “next” for another block, or “end” to close today’s live session.";
      nextState = {
        ...prevState,
        expecting: "next_or_end",
      };
    } else if (prevState.expecting === "next_or_end") {
      if (lower.includes("next")) {
        reply =
          "Next block: repeat a similar structure but slightly adjusted for your energy. (In the full build, this will be fully tailored.) For now, imagine another 2–3 sets in the same style, then type “done” when finished.";
        nextState = {
          ...prevState,
          block: "block2",
          step: (prevState.step || 1) + 1,
          expecting: "done",
        };
      } else if (lower.includes("end")) {
        reply =
          "Session closed. Quick reflection: rate how you feel mentally and physically from 1–10, and name one win from today. When you’re ready, I’ll be here for the next round.";
        nextState = null;
      } else {
        reply =
          'Choose your path: type “next” for another block or “end” to close this live session.';
        nextState = prevState;
      }
    } else {
      reply =
        "Stay locked in. If you’re in the middle of a block, finish it. When it’s done, type “done”. If you want to stop, type “end”.";
      nextState = prevState;
    }
  }

  // -------------------------------------------------------
  // 7. Update memory with new state + minimal fitness info
  // -------------------------------------------------------
  const patch = {
    preferences: {
      ...(memory.preferences || {}),
      fitnessGoal:
        (memory.preferences && memory.preferences.fitnessGoal) ||
        (nextState && nextState.goal) ||
        (prevState && prevState.goal) ||
        null,
    },
    fitness: {
      ...(memory.fitness || {}),
      lastGoal:
        (nextState && nextState.goal) ||
        (prevState && prevState.goal) ||
        memory.fitness?.lastGoal ||
        null,
    },
    live: {
      ...(memory.live || {}),
      state: nextState,
    },
  };

  const updated = await saveUserMemory(userId, patch);

  return {
    reply,
    state: updated.live?.state || null,
  };
}