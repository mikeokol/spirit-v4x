// routes/fitness.js — Spirit v5.0 Training Block Normalization Engine
// ---------------------------------------------------------------
import express from "express";
import OpenAI from "openai";
import supabase from "../supabase.js";

const router = express.Router();
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/*
  This module takes a raw Spirit-generated training plan,
  converts it into structured JSON, then normalizes it
  into a predictable shape:

  {
    day: number,
    title: string,
    focus: string,
    exercises: [
      { name, sets, reps }
    ]
  }
*/

// ==============================
// STEP 1 — Convert plan → JSON
// ==============================
async function extractWorkouts(planText) {
  const system = `
You are Spirit v5.0.

Your task: Convert the following training plan into structured JSON.
Rules:
- Output ONLY JSON. No commentary.
- JSON must be an array of 4–8 objects.
- Each object must contain:
  "day": number
  "title": string
  "focus": string
  "exercises": [
      { "name": string, "sets": string, "reps": string }
  ]
- If unsure about sets/reps, choose safe defaults (3x8–10).
`.trim();

  const user = `
Convert this into JSON days:

${planText}
`.trim();

  try {
    const response = await client.chat.completions.create({
      model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
      messages: [
        { role: "system", content: system },
        { role: "user", content: user }
      ],
      temperature: 0.2,
      max_tokens: 900,
    });

    return JSON.parse(response.choices[0].message.content);
  } catch (err) {
    console.warn("[extractWorkouts] parse failed:", err.message);
    return [];
  }
}

// ==============================
// STEP 2 — Normalize JSON
// ==============================
function normalizeWorkouts(raw = []) {
  const safeDay = (d) => (Number(d) >= 1 && Number(d) <= 7 ? Number(d) : null);

  const output = [];

  for (let i = 1; i <= raw.length; i++) {
    const found =
      raw.find((w) => safeDay(w.day) === i) ||
      raw[i - 1] ||
      null;

    if (!found) {
      output.push({
        day: i,
        title: `Day ${i}`,
        focus: "General Training",
        exercises: [],
      });
      continue;
    }

    output.push({
      day: i,
      title: found.title || `Day ${i}`,
      focus: found.focus || "",
      exercises: Array.isArray(found.exercises)
        ? found.exercises.map((ex) => ({
            name: ex.name || "Exercise",
            sets: ex.sets || "3",
            reps: ex.reps || "8-10",
          }))
        : [],
    });
  }

  return output;
}

// ==============================
// STEP 3 — API Route
// ==============================
router.post("/normalize", async (req, res) => {
  const { userId, planText } = req.body || {};

  if (!userId || !planText) {
    return res.status(400).json({ ok: false, error: "Missing userId or planText" });
  }

  try {
    const raw = await extractWorkouts(planText);
    const normalized = normalizeWorkouts(raw);

    await supabase
      .from("sessions")
      .update({ training_block: { ...normalized } })
      .eq("user_id", userId);

    return res.json({
      ok: true,
      normalized,
    });
  } catch (err) {
    return res.status(500).json({
      ok: false,
      error: "Normalization failed",
      details: err.message,
    });
  }
});

export default router;
