// routes/live.js — Spirit v5.0 Hybrid Live Fitness Coach
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

// Small helper to extract today's structured workout (if present)
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
// POST /live/start
// Start a live coaching session for a given day.
// Body: { userId, day? }
// --------------------------------------------------------------------
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

  const trainingBlock = session.training_block;
  const planText = trainingBlock.plan_text || "";
  const goal = trainingBlock.goal || "not specified";
  const experience = trainingBlock.experience || "not specified";
  const days = trainingBlock.days || "not specified";
  const gender = session.gender || trainingBlock.gender || "unspecified";

  const effectiveDay = day || session.training_day || 1;
  const { todaysWorkout } = getTodaysWorkout(trainingBlock, effectiveDay);

  const systemPrompt = `
You are Spirit v5.0 — a hybrid live coach across Body and Mind.

Identity:
- You are calm, clear, and grounded.
- You speak as the user's future self who trains consistently.
- You combine physical cues with identity and focus, not hype.

Your role for /live/start:
- Welcome the user into today's session.
- Remind them of today's focus in simple language.
- Summarize ONLY the key exercises for Day ${effectiveDay}.
- Include 1–2 short mindset or identity cues (hybrid: body + mind).
- Keep it under ~8 sentences.
- No markdown, no lists, no emojis.
`.trim();

  const userPrompt = `
User profile:
- Goal: ${goal}
- Experience: ${experience}
- Gender: ${gender}
- Training days per week: ${days}

Structured workout object for today (may be null):
${JSON.stringify(todaysWorkout, null, 2)}

Full training block text (fallback context):
${planText}

If todaysWorkout exists, rely on it first and only use planText as backup.
If todaysWorkout is null, infer a reasonable Day ${effectiveDay} summary from the planText.

Guide the user into Day ${effectiveDay} as a live hybrid coach.
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
      "We begin here. Simple session, focused effort, and a clear mind.";

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

// --------------------------------------------------------------------
// POST /live/next-set
// Set-by-set coaching.
// Body: { userId, day, exerciseIndex?, setNumber, difficulty? }
// --------------------------------------------------------------------
router.post("/next-set", async (req, res) => {
  const { userId, day, exerciseIndex = 0, setNumber, difficulty } = req.body || {};

  if (!userId || !day || !setNumber) {
    return res.status(400).json({
      ok: false,
      error: "Missing userId, day, or setNumber",
    });
  }

  const session = await getUserSession(userId);
  if (!session || !session.training_block) {
    return res.json({
      ok: false,
      error: "No training block found. Generate a Fitness plan first.",
    });
  }

  const trainingBlock = session.training_block;
  const goal = trainingBlock.goal || "not specified";
  const experience = trainingBlock.experience || "not specified";
  const gender = session.gender || trainingBlock.gender || "unspecified";

  const effectiveDay = day || session.training_day || 1;
  const { todaysWorkout } = getTodaysWorkout(trainingBlock, effectiveDay);

  const systemPrompt = `
You are Spirit v5.0 — a hybrid live coach across Body and Mind.

Context:
- You are in the middle of a workout.
- The user just clicked "Next Set" for set ${setNumber}.
- They may feel doubt, fatigue, or momentum.

Your role:
- Name the current or next exercise if possible.
- Give 1–2 clear form or tempo cues.
- Add 1 short mindset or identity cue (hybrid: body + mind).
- Mention rest only if needed after the set.
- Adjust tone based on experience level "${experience}" and perceived difficulty "${difficulty || "normal"}".
- 2–4 sentences only.
- No markdown, no lists, no emojis.
`.trim();

  const userPrompt = `
User profile:
- Goal: ${goal}
- Experience: ${experience}
- Gender: ${gender}

Today's day: ${effectiveDay}
Set number: ${setNumber}
Perceived difficulty so far: ${difficulty || "normal"}
Exercise index (hint): ${exerciseIndex}

Structured workout for today (may be null):
${JSON.stringify(todaysWorkout, null, 2)}

If todaysWorkout exists:
- Assume the user is on the main lift or an accessory.
- Use exercise names from the JSON when it makes sense.
If it is null:
- Give a grounded coaching cue appropriate for this goal and experience.

Now give set-specific guidance for this exact set.
`.trim();

  try {
    const completion = await client.chat.completions.create({
      model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: Number(process.env.SPIRIT_TEMPERATURE || 0.6),
      max_tokens: 260,
    });

    const reply =
      completion.choices?.[0]?.message?.content?.trim() ||
      "Lock in for this set. Clean reps, steady breathing, and a clear intention.";

    return res.json({
      ok: true,
      mode: "live_next_set",
      day: effectiveDay,
      setNumber,
      reply,
    });
  } catch (err) {
    console.error("[Spirit /live/next-set error]", err.message);
    return res.status(500).json({
      ok: false,
      error: "Spirit could not coach this set.",
      details: err.message,
    });
  }
});

// --------------------------------------------------------------------
// POST /live/coach
// Mid-session coaching / between sets.
// Body: { userId, day, message }
// --------------------------------------------------------------------
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

  const trainingBlock = session.training_block;
  const planText = trainingBlock.plan_text || "";
  const goal = trainingBlock.goal || "not specified";
  const experience = trainingBlock.experience || "not specified";
  const gender = session.gender || trainingBlock.gender || "unspecified";
  const effectiveDay = day || session.training_day || 1;

  const { todaysWorkout } = getTodaysWorkout(trainingBlock, effectiveDay);

  const systemPrompt = `
You are Spirit v5.0 — a hybrid live coach across Body and Mind.

Context:
- You are in the middle of a workout session.
- The user is talking to you between sets.

Your role:
- Answer their message with short, grounded guidance.
- Blend physical advice and mindset support.
- Adjust difficulty or suggest swaps if needed.
- Never overwhelm beginners.
- Reference specific exercises or today's focus only when it helps.
- 1–4 sentences.
- No markdown, no emojis, no lists.
`.trim();

  const userPrompt = `
User profile:
- Goal: ${goal}
- Experience: ${experience}
- Gender: ${gender}

Today's day: ${effectiveDay}

Structured workout object for today (may be null):
${JSON.stringify(todaysWorkout, null, 2)}

Full plan text (fallback):
${planText}

User message:
"${message}"

Respond as a hybrid live coach. If the workout JSON exists, you may reference it.
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
      "Breathe between sets. Stay honest about your effort and keep the next set simple and focused.";

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

// --------------------------------------------------------------------
// POST /live/complete
// Finish a session, log difficulty, advance training_day.
// Body: { userId, day, difficulty?, notes? }
// --------------------------------------------------------------------
router.post("/complete", async (req, res) => {
  const { userId, day, difficulty, notes } = req.body || {};

  if (!userId || !day) {
    return res.status(400).json({ ok: false, error: "Missing userId or day" });
  }

  await completeDay({ userId, day, difficulty });

  const systemPrompt = `
You are Spirit v5.0 — close the workout as a hybrid coach.

Your role:
- Acknowledge the effort honestly.
- Reflect one thing they did well (even if it must stay general).
- Give 1 simple focus for the next session.
- End with a short identity-based line.
- 4–6 sentences.
- No markdown, no emojis, no lists.
`.trim();

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
      "Session complete. You kept a promise to yourself today. Your next step is simple: show up again and repeat this level of honesty.";

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
