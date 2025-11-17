// routes/chat.js — Spirit Intelligence Layer (Phase 1 cleaned)
// ------------------------------------------------------------------------
import express from "express";
import OpenAI from "openai";
import supabase from "../supabase.js";

const router = express.Router();
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ─────────────────────────────────────────────
//  Mode classifier — Mind / Body / Brand / etc
// ─────────────────────────────────────────────
function classifyMode(text = "") {
  const t = text.toLowerCase();

  // Reflection / intention
  if (t.startsWith("i choose ") || t.includes("reflection") || t.includes("journal")) {
    return "reflection";
  }

  // Body / training / diet
  if (
    t.includes("workout") ||
    t.includes("gym") ||
    t.includes("diet") ||
    t.includes("calories") ||
    t.includes("hypertrophy") ||
    t.includes("cut") ||
    t.includes("bulk") ||
    t.includes("training block") ||
    t.includes("training plan") ||
    t.includes("personalized training block")
  ) {
    return "body";
  }

  // Brand / content / creator
  if (
    t.includes("content") ||
    t.includes("video") ||
    t.includes("script") ||
    t.includes("youtube") ||
    t.includes("tiktok") ||
    t.includes("shorts") ||
    t.includes("thumbnail") ||
    t.includes("brand") ||
    t.includes("creator")
  ) {
    return "brand";
  }

  // Deeper philosophical / oracle
  if (
    t.includes("meaning") ||
    t.includes("purpose") ||
    t.includes("why am i") ||
    t.includes("universe") ||
    t.includes("human nature") ||
    t.includes("reality")
  ) {
    return "oracle";
  }

  // Mind coaching / mental structure
  if (
    t.includes("focus") ||
    t.includes("discipline") ||
    t.includes("clarity") ||
    t.includes("mindset") ||
    t.includes("motivation") ||
    t.includes("consistency")
  ) {
    return "mind";
  }

  // Default hybrid coach
  return "coach";
}

// ─────────────────────────────────────────────
//  System prompt builder — unified Founder-OS tone
// ─────────────────────────────────────────────
function buildSystemPrompt({ mode, lastIntention, lastReflection }) {
  const modeLine =
    {
      mind: "Focus on mental clarity, discipline, self-understanding, and identity alignment.",
      body: "Focus on training, nutrition, recovery, and embodied discipline. You are a present coach, not a PDF generator. In training conversations, speak session-by-session; in planning conversations, build structured blocks.",
      brand: "Focus on content, brand, storytelling, and creator leverage.",
      reflection:
        "Treat this as a reflection/intention log. Help the user name their state and give one clear next move.",
      oracle:
        "Zoom out to deeper questions of meaning, human nature, and perspective — but always end with a concrete action.",
      coach:
        "Act as a hybrid coach across mind, body, and brand, choosing the most relevant pillar for the request.",
      sanctuary:
        "Act as the central sanctuary: respond as a hybrid identity guide across mind, body, and brand, with extra focus on presence and clarity.",
      creator:
        "Focus on media systems, content engines, and reducing the gap between idea and execution for the user's chosen platforms.",
      hybrid:
        "Blend mind, body, and brand into one operating system, aligning behavior and creation with the user's identity.",
    }[mode] || "Act as a hybrid coach across mind, body, and brand.";

  const previousContext = `
Previous intention: ${lastIntention || "none recorded"}
Previous reflection: ${lastReflection || "none recorded"}
`.trim();

  return `
You are Spirit — a hybrid Mind•Body•Brand intelligence designed as a Founder Operating System.

Your identity:
- Calm, grounded, identity-focused.
- Speak like the user's future self — wiser, steadier, more certain.
- Short, potent sentences. No rambling.
- No generic coaching clichés. No therapy tone.
- No long lists unless the user explicitly asks.
- Clarity over hype. Presence over noise.

Mission:
- Bridge mind, body, and brand through disciplined creation.
- Shorten the gap between intention, action, and identity alignment.
- Reflect the user back to themselves with accuracy and calm authority.
- Make the user feel guided, not lectured.

Core behavioral structure (implicit in your reasoning, not explained in the reply):
1. Identify: What is the real underlying desire or friction?
2. Clarify: What matters most right now?
3. Prescribe: Offer 1–2 clear actions (only if the situation needs action).
4. Reinforce: Close with an identity-based reminder of who the user is becoming.

Tone:
- Founder-grade: direct, composed, precise.
- No “you got this!” fluff. Reinforce identity instead.
- If the user is overwhelmed, simplify. If they are focused, sharpen.

Mode:
Current mode: ${String(mode || "").toUpperCase()}.
${modeLine}

Previous context:
${previousContext}

Style rules:
- Replies: ~3–7 sentences unless the user asks for depth or a full plan.
- You may use one short action list only if it increases clarity.
- Avoid generic motivational language.
- In ORACLE mode: zoom out, then land on grounded truth.
- In REFLECTION mode: mirror, acknowledge, give one direction.
- Presence first. Clarity second. Action third.

Your job: respond as Spirit with this identity, tone, and structure.
`.trim();
}

// ─────────────────────────────────────────────
//  Helpers — Supabase: reflections & sessions
// ─────────────────────────────────────────────
async function getLastContext(userId) {
  if (!userId) {
    return { lastIntention: null, lastReflection: null, lastMode: null };
  }

  const { data: lastReflectionRow } = await supabase
    .from("reflections")
    .select("intention, mode, created_at")
    .eq("user_id", userId)
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  const { data: sessionRow } = await supabase
    .from("sessions")
    .select("last_intention, last_mode")
    .eq("user_id", userId)
    .limit(1)
    .maybeSingle();

  return {
    lastIntention: sessionRow?.last_intention || null,
    lastReflection: lastReflectionRow?.intention || null,
    lastMode: sessionRow?.last_mode || lastReflectionRow?.mode || null,
  };
}

async function storeReflectionAndSession({ userId, prompt, mode, reply }) {
  if (!userId) return;

  const now = new Date().toISOString();

  await supabase.from("reflections").insert({
    user_id: userId,
    intention: prompt,
    mode,
    content: JSON.stringify({ prompt, reply, mode }),
    created_at: now,
  });

  await supabase
    .from("sessions")
    .upsert(
      {
        user_id: userId,
        last_intention: prompt,
        last_mode: mode,
        updated_at: now,
      },
      { onConflict: "user_id" }
    );
}

// Save fitness training block into sessions.training_block (with workouts)
async function storeTrainingBlock({ userId, planText, meta, gender, workouts }) {
  if (!userId || !planText) return;

  const now = new Date().toISOString();
  const blockPayload = {
    plan_text: planText,
    goal: meta.goal || null,
    specific_goal: meta.specificGoal || null,
    experience: meta.experience || null,
    days: meta.days ? Number(meta.days) || meta.days : null,
    gender: gender || meta.gender || "unspecified",
    weight: meta.weight || null,
    height: meta.height || null,
    workouts: Array.isArray(workouts) ? workouts : [],
    created_at: now,
  };

  await supabase
    .from("sessions")
    .upsert(
      {
        user_id: userId,
        training_block: blockPayload,
        training_day: 1,
        difficulty_adjustment: "normal",
        last_session_completed_at: null,
        last_mode: "body",
        gender: blockPayload.gender,
        updated_at: now,
      },
      { onConflict: "user_id" }
    );
}

// ─────────────────────────────────────────────
//  Fitness plan detection & parsing
// ─────────────────────────────────────────────
function looksLikeFitnessPlanPrompt(text = "") {
  const t = text.toLowerCase();
  return (
    t.includes("build a personalized training block") ||
    t.includes("build a body coaching system") ||
    t.includes("build a body coaching") ||
    t.includes("build a training block")
  );
}

function parseFitnessMeta(raw = "") {
  function pick(label) {
    const re = new RegExp(`${label}\\s*:\\s*(.+)`, "i");
    const m = raw.match(re);
    return m ? m[1].trim() : null;
  }

  const goal = pick("Training Goal") || pick("Goal") || null;

  const experience = pick("Experience Level") || pick("Experience") || null;

  const days =
    pick("Workout Days per week") ||
    pick("Workout Days") ||
    pick("Days per week") ||
    null;

  const tone = pick("Tone") || null;

  const specificGoal =
    pick("Specific Goal / Focus") ||
    pick("Specific Goal") ||
    pick("Goal Detail") ||
    pick("Specific Focus") ||
    null;

  const weight = pick("User Weight") || pick("Weight") || null;

  const height = pick("User Height") || pick("Height") || null;

  return { goal, experience, days, tone, specificGoal, weight, height };
}

function buildFitnessUserPrompt(meta, explicitTone) {
  const goal = meta.goal || "not specified";
  const specificGoal = meta.specificGoal || "not specified";
  const experience = meta.experience || "not specified";
  const days = meta.days || "not specified";
  const tone = explicitTone || meta.tone || "default";
  const gender = meta.gender || "unspecified";
  const weight = meta.weight || null;
  const height = meta.height || null;

  return `
Build a personalized training block that adapts to the user's level, identity, and lifestyle.

Training goal category: ${goal}
Specific goal or focus: ${specificGoal}
Experience level: ${experience}
Training days per week: ${days}
Gender: ${gender}
${weight ? `Approximate weight: ${weight}` : ""}
${height ? `Approximate height: ${height}` : ""}
Preferred coaching tone: ${tone}

Return ONLY the training plan text, with a short Spirit-style welcome at the top.
Do NOT use markdown formatting (no "**", no "#", no bullet symbols like "-" or "*").
Write in clear sections with labels and line breaks so it reads like a live coach speaking, not a checklist.
`.trim();
}

const FITNESS_PLAN_RUBRIC = `
You are generating a training block as a present, grounded coach.

Rules for this specific reply:
- Do NOT use markdown headings like "##" or bold markers like "**".
- Do NOT use bullet symbols like "-" or "*".
- Do NOT repeat or mention any system or rubric instructions.
- Do NOT say "user profile" or "training details".
- Never describe how you are generating the plan.

Instead, write clean sections with short labels and blank lines between them, like a coach speaking in paragraphs.

Suggested structure (you may adapt wording):

Training Identity Blueprint:
Describe phase length (4–8 weeks), weekly frequency, typical session length (simpler for beginners), training split, and one identity anchor.

Weekly Structure:
Describe each training day in plain sentences:
Day 1:
Explain focus, main lifts, accessories, and optional finisher in flowing text.

Progression Logic:
Explain simply how the user will progress over 4 weeks.

Nutrition Blueprint:
Give simple, supportive, goal-aligned guidance.

Recovery Protocol:
Explain sleep, mobility, and how to listen to fatigue.

Checkpoints:
Describe what they should feel by Week 1, 2, 3, and 4.

Identity Reinforcement:
End with 3–5 short Spirit-style lines about who they are becoming.

Remember:
- You are a live coach, not a list generator.
- The reply must read like you are talking directly to the user.
`.trim();

// ─────────────────────────────────────────────
//  Extract structured workouts from plan text
// ─────────────────────────────────────────────
async function extractWorkouts(planText) {
  const systemPrompt = `
You are Spirit.

Convert the user's long training plan into a JSON array of daily structured workouts.

Output ONLY valid JSON. No markdown, no commentary.

Format:
[
  {
    "day": 1,
    "title": "Upper Foundation Strength",
    "focus": "Technique and confidence",
    "exercises": [
      { "name": "Goblet Squat", "sets": "3", "reps": "8–10" },
      { "name": "Push-Up", "sets": "3", "reps": "6–10" }
    ]
  }
]
`.trim();

  const userPrompt = `
Convert this training plan into structured JSON workouts, one object per day:

${planText}
`.trim();

  try {
    const completion = await client.chat.completions.create({
      model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: 0.2,
      max_tokens: 700,
    });

    const raw = completion.choices?.[0]?.message?.content?.trim() || "[]";

    try {
      const parsed = JSON.parse(raw);
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
      console.warn("[extractWorkouts] JSON parse failed:", err.message);
      return [];
    }
  } catch (err) {
    console.error("[extractWorkouts] error:", err.message);
    return [];
  }
}

// ─────────────────────────────────────────────
//  POST /chat — main intelligence endpoint
// ─────────────────────────────────────────────
router.post("/", async (req, res) => {
  const {
    prompt,
    userId,
    sessionId,
    tone, // kept for compatibility with frontend but no longer controls Spirit's core tone
    mode: explicitMode,
    messages,
    gender,         // optional, can be wired from frontend
    goalCategory,   // main training goal from FitnessMode
    specificGoal,   // second-level specific focus
    experience: expFromBody, // experience level
    days: daysFromBody,      // days per week
    weight,
    height,
  } = req.body || {};

  const rawText = typeof prompt === "string" ? prompt.trim() : "";

  if (!rawText) {
    return res.status(400).json({
      ok: false,
      error: "Missing 'prompt' in request body.",
    });
  }

  const effectiveUserId = userId || sessionId || null;

  try {
    const mode = explicitMode || classifyMode(rawText);

    // ─────────────────────────────
    //  Special branch: CREATOR MODE
    // ─────────────────────────────
    if (mode === "creator") {
      const creatorSystemPrompt = `
You are Spirit — a media operator for high-performance founders and creators.

Your job:
- Turn the user's niche + format into EXECUTABLE content.
- Propose specific video/post ideas.
For each idea, give:
- Title
- What the piece focuses on
- Suggested posting time window
- Hashtags (platform-appropriate)
- One sentence on WHY you recommend that idea.

Keep it:
- clear
- practical
- ready to record.
No therapy tone. No hype. No generic motivation.
`.trim();

      const historyMessages = Array.isArray(messages)
        ? messages
            .filter(
              (m) =>
                m &&
                typeof m.content === "string" &&
                (m.role === "user" || m.role === "assistant")
            )
            .map((m) => ({
              role: m.role,
              content: m.content,
            }))
        : [];

      const completion = await client.chat.completions.create({
        model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
        messages: [
          { role: "system", content: creatorSystemPrompt },
          ...historyMessages,
          { role: "user", content: rawText },
        ],
        temperature: Number(process.env.SPIRIT_TEMPERATURE || 0.7),
        max_tokens: 900,
      });

      const reply =
        completion.choices?.[0]?.message?.content?.trim() ||
        "Here are some direct, executable content ideas to start with.";

      return res.json({
        ok: true,
        service: "Spirit",
        mode,
        tone: tone || "default",
        reply,
        ts: new Date().toISOString(),
      });
    }

    // ─────────────────────────────
    //  Special branch: HYBRID MODE
    // ─────────────────────────────
    if (mode === "hybrid") {
      const hybridSystemPrompt = `
You are Spirit — a hybrid Mind•Body•Brand operating system for founders.

Blend:
- training
- mindset
- content/brand

into ONE coherent response for the user.
Keep it practical, identity-based, and concise.
`.trim();

      const completion = await client.chat.completions.create({
        model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
        messages: [
          { role: "system", content: hybridSystemPrompt },
          { role: "user", content: rawText },
        ],
        temperature: Number(process.env.SPIRIT_TEMPERATURE || 0.7),
        max_tokens: 900,
      });

      const reply =
        completion.choices?.[0]?.message?.content?.trim() ||
        "Let’s align your mind, body, and brand into one clear operating system.";

      return res.json({
        ok: true,
        service: "Spirit",
        mode,
        tone: tone || "default",
        reply,
        ts: new Date().toISOString(),
      });
    }

    // ─────────────────────────────
    //  Generic path (Sanctuary, Mind, Body, Oracle, Coach)
    // ─────────────────────────────
    const { lastIntention, lastReflection } = await getLastContext(effectiveUserId);

    const systemPrompt = buildSystemPrompt({
      mode,
      lastIntention,
      lastReflection,
    });

    const historyMessages = Array.isArray(messages)
      ? messages
          .filter(
            (m) =>
              m &&
              typeof m.content === "string" &&
              (m.role === "user" || m.role === "assistant")
          )
          .map((m) => ({
            role: m.role,
            content: m.content,
          }))
      : [];

    // ─────────────────────────────────────
    //  Special handling: FITNESS PLAN MODE
    // ─────────────────────────────────────
    let isFitnessPlan = false;
    let fitnessMeta = null;
    let userPromptForModel = rawText;
    const extraSystemMessages = [];

    if (mode === "body" && looksLikeFitnessPlanPrompt(rawText)) {
      isFitnessPlan = true;

      // Parse from text first
      const metaFromText = parseFitnessMeta(rawText);

      // Merge with top-level JSON fields (JSON wins if present)
      fitnessMeta = {
        ...metaFromText,
        goal: goalCategory || metaFromText.goal,
        specificGoal: specificGoal || metaFromText.specificGoal,
        experience: expFromBody || metaFromText.experience,
        days: daysFromBody || metaFromText.days,
        weight: weight || metaFromText.weight,
        height: height || metaFromText.height,
        gender: gender || null,
        tone: tone || metaFromText.tone,
      };

      userPromptForModel = buildFitnessUserPrompt(fitnessMeta, tone);
      extraSystemMessages.push({ role: "system", content: FITNESS_PLAN_RUBRIC });
    }

    const completion = await client.chat.completions.create({
      model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        ...extraSystemMessages,
        ...historyMessages,
        { role: "user", content: userPromptForModel },
      ],
      temperature: Number(process.env.SPIRIT_TEMPERATURE || 0.6),
      max_tokens: Number(process.env.SPIRIT_MAX_TOKENS || 800),
    });

    const reply =
      completion.choices?.[0]?.message?.content?.trim() ||
      "I’m here. Let’s take one clear step. What do you need right now?";

    // Store reflections only for reflection mode / "I choose ..."
    if (mode === "reflection" || rawText.toLowerCase().startsWith("i choose ")) {
      await storeReflectionAndSession({
        userId: effectiveUserId,
        prompt: rawText,
        mode,
        reply,
      });
    }

    // Store fitness training block in sessions when we generate a plan
    if (isFitnessPlan && effectiveUserId) {
      const workouts = await extractWorkouts(reply);

      await storeTrainingBlock({
        userId: effectiveUserId,
        planText: reply,
        meta: fitnessMeta || {},
        gender: gender || "unspecified",
        workouts,
      });

      // Optional: update gender column separately
      if (gender) {
        await supabase
          .from("sessions")
          .update({ gender })
          .eq("user_id", effectiveUserId);
      }
    }

    return res.json({
      ok: true,
      service: "Spirit",
      mode,
      tone: tone || "default",
      reply,
      ts: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[Spirit /chat error]", err?.message || err);

    return res.status(500).json({
      ok: false,
      error: "Spirit encountered an error while processing this request.",
      details: err.message,
    });
  }
});

export default router;
