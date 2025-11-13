// routes/chat.js — Spirit v4.x Intelligence Layer
// ------------------------------------------------
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
    t.includes("bulk")
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
    t.includes("brand")
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

  // Default hybrid
  return "coach";
}

// ─────────────────────────────────────────────
//  System prompt builder — Spirit v4.x brain
// ─────────────────────────────────────────────
function buildSystemPrompt({ mode, tone, lastIntention, lastReflection }) {
  let toneDescriptor = "";
  switch (tone) {
    case "mystical":
      toneDescriptor =
        "Use a calm, mystical, presence-first tone, like an oracle that respects time and clarity.";
      break;
    case "high-performance":
      toneDescriptor =
        "Use a high-performance, elite-athlete coach tone: direct, focused, no fluff.";
      break;
    case "casual":
      toneDescriptor =
        "Use a relaxed, conversational tone, but still precise and actionable.";
      break;
    case "founder":
      toneDescriptor =
        "Use a founder-to-founder tone: strategic, brutally honest, but encouraging.";
      break;
    default:
      toneDescriptor =
        "Use Spirit's default tone: mystical, disciplined, supportive, concise, and action-focused.";
  }

  const modeLine = {
    mind: "Focus on mental clarity, discipline, self-understanding, and identity alignment.",
    body: "Focus on training, nutrition, recovery, and embodied discipline.",
    brand: "Focus on content, brand, storytelling, and creator leverage.",
    reflection:
      "Treat this as a reflection/intention log. Help the user name their state and give one clear next move.",
    oracle:
      "Zoom out to deeper questions of meaning, human nature, and perspective — but always end with a concrete action.",
    coach:
      "Act as a hybrid coach across mind, body, and brand, choosing the most relevant pillar for the request.",
  }[mode];

  const previousContext = `
Previous intention: ${lastIntention || "none recorded"}
Previous reflection: ${lastReflection || "none recorded"}
`.trim();

  return `
You are Spirit v4.x — a hybrid Mind•Body•Brand coach and creator engine.

Mission:
- Bridge mind, body, and brand through disciplined creation.
- Shorten the gap between idea, plan, and execution.
- Make the user feel guided, not lectured.

Operating rules:
1. ALWAYS follow this structure in your thinking (you can keep labels implicit in the reply):
   - Identify: What is the real problem / desire?
   - Clarify: What matters most in this moment?
   - Prescribe: Give 1–3 specific, realistic actions the user can take now or today.
   - Reinforce: Close with a short, identity-based reinforcement (who they are becoming).

2. Responses must be:
   - Concise (3–8 sentences total, unless the user explicitly asks for depth).
   - Actionable (at least one concrete step).
   - Non-generic and directly tied to the prompt.
   - No filler like "As an AI".

3. Use context if available:
   ${previousContext}

4. Mode:
   - Current mode: ${mode.toUpperCase()}.
   - ${modeLine}

5. Tone:
   - ${toneDescriptor}

6. Style:
   - Avoid long bullet lists unless requested.
   - You may use a short list for actions if it improves clarity.
   - Slightly oracular, but never vague. Mystical clarity, not mystic fog.

Your job: respond as Spirit in this mode, with this structure and tone.
`.trim();
}

// ─────────────────────────────────────────────
//  Supabase helpers — reflections & sessions
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

// ─────────────────────────────────────────────
//  POST /chat — main intelligence endpoint
// ─────────────────────────────────────────────
router.post("/", async (req, res) => {
  const { prompt, userId, sessionId, tone } = req.body || {};
  const text = typeof prompt === "string" ? prompt.trim() : "";

  if (!text) {
    return res.status(400).json({
      ok: false,
      error: "Missing 'prompt' in request body.",
    });
  }

  // Option A: per-session UUID comes from frontend
  const effectiveUserId = userId || sessionId || null;

  try {
    const mode = classifyMode(text);

    const { lastIntention, lastReflection } = await getLastContext(effectiveUserId);

    const systemPrompt = buildSystemPrompt({
      mode,
      tone,
      lastIntention,
      lastReflection,
    });

    const completion = await client.chat.completions.create({
      model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: text },
      ],
      temperature: 0.6,
      max_tokens: 400,
    });

    const reply =
      completion.choices?.[0]?.message?.content?.trim() ||
      "I’m here. Let’s take one clear step. What do you need right now?";

    if (mode === "reflection" || text.toLowerCase().startsWith("i choose ")) {
      await storeReflectionAndSession({
        userId: effectiveUserId,
        prompt: text,
        mode,
        reply,
      });
    }

    return res.json({
      ok: true,
      service: "Spirit v4.x",
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