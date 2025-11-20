// routes/chat/FitnessEngine.js
// ---------------------------------------------------------------------------
// Spirit v5.1 — Fitness Intelligence Engine
// Handles:
//  • Fitness prompt detection
//  • Metadata parsing
//  • Building the training-block prompt
//  • Extracting structured workouts (JSON)
// ---------------------------------------------------------------------------

import OpenAI from "openai";

// ---------------------------------------------------------------------------
// Detect whether the incoming message relates to training
// ---------------------------------------------------------------------------
export function detectFitnessPrompt(text = "") {
  const t = text.toLowerCase();

  return (
    t.includes("training goal") ||
    t.includes("build a training block") ||
    t.includes("build a body coaching") ||
    t.includes("training block") ||
    t.includes("workout days") ||
    t.includes("experience level")
  );
}

// ---------------------------------------------------------------------------
// Extract key fitness fields from user text (fallback if UI doesn't send them)
// ---------------------------------------------------------------------------
export function parseFitnessMeta(raw = "") {
  function pick(label) {
    const re = new RegExp(`${label}\\s*:\\s*(.+)`, "i");
    const m = raw.match(re);
    return m ? m[1].trim() : null;
  }

  return {
    goal: pick("Training Goal") || pick("Goal") || null,
    experience: pick("Experience Level") || pick("Experience") || null,
    days: pick("Workout Days") || pick("Days per week") || null,
    tone: pick("Tone") || null,
    specificGoal:
      pick("Specific Goal") ||
      pick("Specific Focus") ||
      pick("Goal Detail") ||
      null,
    weight: pick("User Weight") || pick("Weight") || null,
    height: pick("User Height") || pick("Height") || null,
  };
}

// ---------------------------------------------------------------------------
// Build the user-facing prompt for generating a training block
// ---------------------------------------------------------------------------
export function buildFitnessPrompt(meta) {
  return `
Build a personalized training block that adapts to the user's identity, experience, and lifestyle.

Training goal: ${meta.goal || "not specified"}
Specific focus: ${meta.specificGoal || "not specified"}
Experience level: ${meta.experience || "not specified"}
Training days per week: ${meta.days || "not specified"}
Gender: ${meta.gender || "unspecified"}
${meta.weight ? `Approximate weight: ${meta.weight}` : ""}
${meta.height ? `Approximate height: ${meta.height}` : ""}
Preferred tone: ${meta.tone || "default"}

Return ONLY the training plan text. Do not include markdown. 
Speak like a real coach in clean labeled sections.
  `.trim();
}

// ---------------------------------------------------------------------------
// Convert the long training plan text into JSON Daily Workouts
// ---------------------------------------------------------------------------
export async function extractWorkouts(planText, model, apiKey) {
  const client = new OpenAI({ apiKey });

  const systemPrompt = `
You are Spirit.

Convert the user's full training plan into a JSON array of daily structured workouts.

Output ONLY valid JSON. No markdown. No comments.

Format example:
[
  {
    "day": 1,
    "title": "Upper Foundation Strength",
    "focus": "Technique and confidence",
    "exercises": [
      { "name": "Goblet Squat", "sets": "3", "reps": "8–10" }
    ]
  }
]
  `.trim();

  const userPrompt = `
Convert this training plan into structured JSON workouts:

${planText}
  `.trim();

  try {
    const completion = await client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: 0.2,
      max_tokens: 700,
    });

    const raw = completion.choices?.[0]?.message?.content?.trim() || "[]";
    const parsed = JSON.parse(raw);

    // Validate and normalize output
    if (Array.isArray(parsed)) {
      return parsed.map((w, idx) => ({
        day: Number(w.day) || idx + 1,
        title: w.title || `Day ${idx + 1}`,
        focus: w.focus || "",
        exercises: Array.isArray(w.exercises) ? w.exercises : [],
      }));
    }

    return [];
  } catch (err) {
    console.warn("[extractWorkouts] JSON parsing failed:", err.message);

    // Safe fallback
    return [
      {
        day: 1,
        title: "Full Body Foundation",
        focus: "Movement and consistency",
        exercises: [],
      },
    ];
  }
}