// routes/live.js — Spirit v4.x Live Fitness Coach (v1)
// ----------------------------------------------------
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

  // Increment training_day (simple v1 logic)
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

  // Also log into training_history (if table exists)
  try {
    await supabase.from("training_history").insert({
      user_id: userId,
      session_day: day,
      block_week: null,
      perceived_difficulty: difficulty || "normal",
      notes: null,
    });
  } catch (err) {
    console.warn("[live/completeDay] training_history insert failed:", err.message);
  }
}

// ----------------------------------------------------
// POST /live/start
// Start a live coaching session for a given day.
// Body: { userId, day? }
// ----------------------------------------------------
router.post("/start", async (req, res) => {
  const { userId, day } = req.body || {};

  if (!userId) {
    return res.status(400).json({ ok: false, error: "Missing userId" });
  }

  const session = await getUserSession(userId);
  if (!session || !session.training_block) {
    return res.json({
      ok: false,
      error: "No training block found. Generate a Fitness plan first.",
    });
  }

  const plan = session.training_block.plan_text || "";
  const goal = session.training_block.goal || "not specified";
  const experience = session.training_block.experience || "not specified";
  const days = session.training_block.days || "not specified";
  const gender = session.gender || "unspecified";

  const effectiveDay = day || session.training_day || 1;

  const systemPrompt = `
You are Spirit v4.x — a live fitness coach.

You are guiding a real-time workout session, not generating a new plan.
You already have the user's full training block.

Your job in this message:
- Welcome the user to today's session
- Briefly remind them of today's focus
- Summarize ONLY the exercises for Day ${effectiveDay}
- Keep it beginner-friendly if experience is beginner
- Tone: grounded, encouraging, identity-focused
- Keep it under ~10 sentences.
`;

  const userPrompt = `
User profile:
- Goal: ${goal}
- Experience: ${experience}
- Gender: ${gender}
- Training days per week: ${days}

Full training block (for context):
${plan}

Now, guide the user into Day ${effectiveDay} as a live coach.
`.trim();

  try {
    const completion = await client.chat.completions.create({
      model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: Number(process.env.SPIRIT_TEMPERATURE || 0.6),
      max_tokens: 400,
    });

    const reply =
      completion.choices?.[0]?.message?.content?.trim() ||
      "We begin. Today we move with simplicity and focus.";

    return res.json({
      ok: true,
      mode: "live_start",
      day: effectiveDay,
      reply,
    });
  } catch (err) {
    console.error("[Spirit /live/start error]", err.message);
    return res.status(500).json({
      ok: false,
      error: "Spirit could not start the live session.",
      details: err.message,
    });
  }
});

// ----------------------------------------------------
// POST /live/coach
// Mid-session coaching / between sets.
// Body: { userId, day, message }
// ----------------------------------------------------
router.post("/coach", async (req, res) => {
  const { userId, day, message } = req.body || {};

  if (!userId || !message) {
    return res.status(400).json({ ok: false, error: "Missing userId or message" });
  }

  const session = await getUserSession(userId);
  if (!session || !session.training_block) {
    return res.json({
      ok: false,
      error: "No training block found. Generate a Fitness plan first.",
    });
  }

  const plan = session.training_block.plan_text || "";
  const goal = session.training_block.goal || "not specified";
  const experience = session.training_block.experience || "not specified";
  const gender = session.gender || "unspecified";
  const effectiveDay = day || session.training_day || 1;

  const systemPrompt = `
You are Spirit v4.x — a live fitness coach.

You are in the middle of a workout session.
Respond to the user's message with:
- short guidance (1–4 sentences)
- identity-based encouragement
- if needed, slightly adjust difficulty or give substitutions
- never overwhelm beginners
`;

  const userPrompt = `
User profile:
- Goal: ${goal}
- Experience: ${experience}
- Gender: ${gender}
Current training block:
${plan}

Current session day: ${effectiveDay}
User message: "${message}"
`.trim();

  try {
    const completion = await client.chat.completions.create({
      model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: Number(process.env.SPIRIT_TEMPERATURE || 0.6),
      max_tokens: 250,
    });

    const reply =
      completion.choices?.[0]?.message?.content?.trim() ||
      "Breathe. Keep it simple. One set at a time.";

    return res.json({
      ok: true,
      mode: "live_coach",
      day: effectiveDay,
      reply,
    });
  } catch (err) {
    console.error("[Spirit /live/coach error]", err.message);
    return res.status(500).json({
      ok: false,
      error: "Spirit could not respond as live coach.",
      details: err.message,
    });
  }
});

// ----------------------------------------------------
// POST /live/complete
// Finish a session, log difficulty, advance training_day.
// Body: { userId, day, difficulty?, notes? }
// ----------------------------------------------------
router.post("/complete", async (req, res) => {
  const { userId, day, difficulty, notes } = req.body || {};

  if (!userId || !day) {
    return res.status(400).json({ ok: false, error: "Missing userId or day" });
  }

  await completeDay({ userId, day, difficulty });

  const systemPrompt = `
You are Spirit v4.x — close the workout session.

Your job:
- Acknowledge effort
- Reflect one thing they did well
- Give 1 simple focus for the next session
- End with a short identity-based reminder
Keep it under 6 sentences.
`;

  const userPrompt = `
Day completed: ${day}
Perceived difficulty: ${difficulty || "normal"}
User notes: ${notes || "none"}
`.trim();

  try {
    const completion = await client.chat.completions.create({
      model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: Number(process.env.SPIRIT_TEMPERATURE || 0.6),
      max_tokens: 250,
    });

    const reply =
      completion.choices?.[0]?.message?.content?.trim() ||
      "Session complete. You kept a promise to yourself today.";

    return res.json({
      ok: true,
      mode: "live_complete",
      reply,
    });
  } catch (err) {
    console.error("[Spirit /live/complete error]", err.message);
    return res.status(500).json({
      ok: false,
      error: "Spirit could not close the session.",
      details: err.message,
    });
  }
});

export default router;
