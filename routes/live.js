// routes/live.js — Spirit v5.0 Prep: Clean, Identity-Driven Live Coaching Layer
// --------------------------------------------------------------------
import express from "express";
import OpenAI from "openai";
import supabase from "../supabase.js";

const router = express.Router();
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Helper: get the user's session row
async function getUserSession(userId) {
  const { data, error } = await supabase
    .from("sessions")
    .select("*")
    .eq("user_id", userId)
    .maybeSingle();

  if (error) {
    console.error("[live/getUserSession] error:", error.message);
    return null;
  }
  return data;
}

// Helper: update session training day + last completed
async function completeDay({ userId, day, difficulty }) {
  const now = new Date().toISOString();

  const { data: current } = await supabase
    .from("sessions")
    .select("training_day")
    .eq("user_id", userId)
    .maybeSingle();

  const nextDay = (current?.training_day || 1) + 1;

  await supabase
    .from("sessions")
    .update({
      training_day: nextDay,
      last_session_completed_at: now,
      difficulty_adjustment: difficulty || "normal",
      updated_at: now,
    })
    .eq("user_id", userId);

  // Optional history logging
  try {
    await supabase.from("training_history").insert({
      user_id: userId,
      session_day: day,
      perceived_difficulty: difficulty || "normal",
      notes: null,
    });
  } catch (err) {
    console.warn("[live/completeDay] training_history insert failed:", err.message);
  }
}

// Extract today's workout
function getTodaysWorkout(trainingBlock, effectiveDay) {
  if (!trainingBlock) return { todaysWorkout: null, workouts: [] };

  const workouts = Array.isArray(trainingBlock.workouts)
    ? trainingBlock.workouts
    : [];

  const todaysWorkout =
    workouts.find((w) => Number(w.day) === Number(effectiveDay)) || null;

  return { todaysWorkout, workouts };
}

// --------------------------------------------------------------------
// POST /live/start — Begin session
// --------------------------------------------------------------------
router.post("/start", async (req, res) => {
  const { userId, day } = req.body || {};
  if (!userId) return res.status(400).json({ ok: false, error: "Missing userId" });

  const session = await getUserSession(userId);
  if (!session || !session.training_block) {
    return res.json({
      ok: false,
      error: "No training block found. Build a training system first.",
    });
  }

  const block = session.training_block;
  const effectiveDay = day || session.training_day || 1;
  const { todaysWorkout } = getTodaysWorkout(block, effectiveDay);

  const systemPrompt = `
You are Spirit — a live training guide with a calm, grounded, identity-driven voice.
You do NOT explain systems, do not lecture, do not hype.
You speak with clarity, simplicity, and presence.

For a live session:
- Welcome the user
- Name today's focus
- Summarize ONLY today's exercises
- 5–7 sentences max
- Identity-first, not motivational fluff
`.trim();

  const userPrompt = `
User:
Goal: ${block.goal}
Experience: ${block.experience}
Days/week: ${block.days}
Gender: ${session.gender || block.gender || "unspecified"}

Today's structured workout:
${JSON.stringify(todaysWorkout, null, 2)}

Full block (fallback):
${block.plan_text}

Guide the user into Day ${effectiveDay} with clarity.
`.trim();

  try {
    const c = await client.chat.completions.create({
      model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
      messages: [{ role: "system", content: systemPrompt }, { role: "user", content: userPrompt }],
      temperature: Number(process.env.SPIRIT_TEMPERATURE || 0.5),
      max_tokens: 380,
    });

    const reply = c.choices?.[0]?.message?.content?.trim() || "We begin. Breathe in your direction.";
    return res.json({ ok: true, mode: "live_start", day: effectiveDay, reply });
  } catch (err) {
    console.error("[Spirit /live/start error]", err.message);
    return res.status(500).json({ ok: false, error: "Could not start session.", details: err.message });
  }
});

// --------------------------------------------------------------------
// POST /live/next-set — Set-by-set coaching
// --------------------------------------------------------------------
router.post("/next-set", async (req, res) => {
  const { userId, day, exerciseIndex = 0, setNumber, difficulty } = req.body || {};
  if (!userId || !day || !setNumber)
    return res.status(400).json({ ok: false, error: "Missing userId, day, or setNumber" });

  const session = await getUserSession(userId);
  if (!session || !session.training_block)
    return res.json({ ok: false, error: "No training block found." });

  const block = session.training_block;
  const effectiveDay = day || session.training_day || 1;
  const { todaysWorkout } = getTodaysWorkout(block, effectiveDay);

  const systemPrompt = `
You are Spirit — live training guidance.

For THIS set:
- Identify the exercise if available.
- Give ONE simple cue.
- Identity-forward tone.
- 2–4 sentences maximum.
`.trim();

  const userPrompt = `
Experience: ${block.experience}
Difficulty: ${difficulty || "normal"}
Day: ${effectiveDay}
Set: ${setNumber}

Workout for today:
${JSON.stringify(todaysWorkout, null, 2)}

Provide set-specific guidance.
`.trim();

  try {
    const c = await client.chat.completions.create({
      model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
      messages: [{ role: "system", content: systemPrompt }, { role: "user", content: userPrompt }],
      temperature: Number(process.env.SPIRIT_TEMPERATURE || 0.45),
      max_tokens: 180,
    });

    const reply = c.choices?.[0]?.message?.content?.trim() || "Steady. This set defines your line.";
    return res.json({ ok: true, mode: "live_next_set", day: effectiveDay, setNumber, reply });
  } catch (err) {
    console.error("[Spirit /live/next-set error]", err.message);
    return res.status(500).json({ ok: false, error: "Could not coach set.", details: err.message });
  }
});

// --------------------------------------------------------------------
// POST /live/coach — Mid-session chat
// --------------------------------------------------------------------
router.post("/coach", async (req, res) => {
  const { userId, day, message } = req.body || {};
  if (!userId || !message)
    return res.status(400).json({ ok: false, error: "Missing userId or message" });

  const session = await getUserSession(userId);
  if (!session || !session.training_block)
    return res.json({ ok: false, error: "No training block found." });

  const block = session.training_block;
  const effectiveDay = day || session.training_day || 1;
  const { todaysWorkout } = getTodaysWorkout(block, effectiveDay);

  const systemPrompt = `
You are Spirit — identity-driven live coaching.

Rules:
- 1–4 sentences
- reference exercises ONLY if helpful
- calm precision
- no hype, no fluff
`.trim();

  const userPrompt = `
User message: "${message}"

Experience: ${block.experience}
Goal: ${block.goal}

Today's workout:
${JSON.stringify(todaysWorkout, null, 2)}

Reply with simple, grounded guidance.
`.trim();

  try {
    const c = await client.chat.completions.create({
      model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
      messages: [{ role: "system", content: systemPrompt }, { role: "user", content: userPrompt }],
      temperature: Number(process.env.SPIRIT_TEMPERATURE || 0.55),
      max_tokens: 180,
    });

    const reply = c.choices?.[0]?.message?.content?.trim() || "Stay inside the rep. One breath at a time.";
    return res.json({ ok: true, mode: "live_coach", day: effectiveDay, reply });
  } catch (err) {
    console.error("[Spirit /live/coach error]", err.message);
    return res.status(500).json({ ok: false, error: "Could not coach mid-session.", details: err.message });
  }
});

// --------------------------------------------------------------------
// POST /live/complete — Finish session
// --------------------------------------------------------------------
router.post("/complete", async (req, res) => {
  const { userId, day, difficulty, notes } = req.body || {};
  if (!userId || !day)
    return res.status(400).json({ ok: false, error: "Missing userId or day" });

  await completeDay({ userId, day, difficulty });

  const systemPrompt = `
You are Spirit — close the session cleanly.
4–6 sentences.
Identity-first.
One focus for next time.
`.trim();

  const userPrompt = `
Day: ${day}
Difficulty: ${difficulty || "normal"}
Notes: ${notes || "none"}
`.trim();

  try {
    const c = await client.chat.completions.create({
      model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
      messages: [{ role: "system", content: systemPrompt }, { role: "user", content: userPrompt }],
      temperature: Number(process.env.SPIRIT_TEMPERATURE || 0.5),
      max_tokens: 200,
    });

    const reply = c.choices?.[0]?.message?.content?.trim() || "Session complete. Quiet discipline moves you forward.";
    return res.json({ ok: true, mode: "live_complete", reply });
  } catch (err) {
    console.error("[Spirit /live/complete error]", err.message);
    return res.status(500).json({ ok: false, error: "Could not close session.", details: err.message });
  }
});

export default router;
