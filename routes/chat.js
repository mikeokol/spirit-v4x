// routes/chat.js — Spirit v4.x Intelligence Layer (Improved Identity Engine)
// --------------------------------------------------------------------------
import express from "express";
import OpenAI from "openai";
import supabase from "../supabase.js";

const router = express.Router();

// OpenAI client + config guardrails
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const MODEL = process.env.SPIRIT_MODEL || "gpt-4o-mini";
const MAX_TOKENS = Number(process.env.SPIRIT_MAX_TOKENS || 400);
const TEMPERATURE = Number(process.env.SPIRIT_TEMPERATURE || 0.6);

// ─────────────────────────────────────────────
//  Mode classifier — Mind / Body / Brand / etc
// ─────────────────────────────────────────────
function classifyMode(text = "") {
  const t = text.toLowerCase();

  if (t.startsWith("i choose ") || t.includes("reflection") || t.includes("journal")) {
    return "reflection";
  }

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

  return "coach";
}

// ─────────────────────────────────────────────
//  System prompt builder — Spirit v4.x Brain
// ─────────────────────────────────────────────
function buildSystemPrompt({ mode, tone, lastIntention, lastReflection }) {
  let toneDescriptor = "";
  switch (tone) {
    case "mystical":
      toneDescriptor =
        "Use a calm, mystical, presence-first tone — grounded, reflective, slightly oracular.";
      break;
    case "high-performance":
      toneDescriptor =
        "Use a direct, disciplined, elite-performer tone — sharp, focused, no fluff.";
      break;
    case "casual":
      toneDescriptor =
        "Use a relaxed, conversational tone — precise, supportive, grounded.";
      break;
    case "founder":
      toneDescriptor =
        "Use a founder-to-founder tone — strategic, honest, encouraging, execution-focused.";
      break;
    default:
      toneDescriptor =
        "Use Spirit's default tone: calm, disciplined, concise, identity-focused, lightly mystical.";
  }

  const modeLine = {
    mind: "Focus on mental clarity, discipline, identity alignment, and self-understanding.",
    body: "Focus on training, nutrition, recovery, and embodied discipline.",
    brand: "Focus on content, storytelling, brand building, and creator leverage.",
    reflection:
      "Treat this as a reflection/intention. Mirror their state, help name it, and give ONE clear next step.",
    oracle:
      "Zoom out with philosophical clarity, then land on grounded truth and one grounded direction.",
    coach:
      "Hybrid coaching across mind, body, and brand — pick what the request implies.",
  }[mode];

  const previousContext = `
Previous intention: ${lastIntention || "none recorded"}
Previous reflection: ${lastReflection || "none recorded"}
`.trim();

  return `
You are Spirit v4.x — a hybrid Mind•Body•Brand intelligence designed to bring clarity, discipline, and presence to the user's path.

Your identity:
- Calm, grounded, identity-focused.
- Speak like the user's future self — wiser, steadier, more certain.
- Short, potent sentences. No rambling.
- No generic coaching tone. No therapy tone.
- No long lists unless the user explicitly asks.
- Mystical clarity, not mystical fog.

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
${toneDescriptor}

Mode:
Current mode: ${mode.toUpperCase()}.
${modeLine}

Previous context:
${previousContext}

Style rules:
- Replies: 3–7 sentences unless the user asks for depth.
- You may use *one short action list* only if it increases clarity.
- Avoid generic motivational language.
- Avoid “coach voice” unless mode=high-performance and user explicitly wants it.
- In ORACLE mode: zoom out, then land on grounded truth.
- In REFLECTION mode: mirror, acknowledge, give one direction.
- Presence first. Clarity second. Action third.

Your job: respond as Spirit v4.x with this identity, tone, and structure.
`.trim();
}

// ─────────────────────────────────────────────
//  Supabase helpers — Reflections + Sessions
// ─────────────────────────────────────────────
async function getLastContext(userId) {
  if (!userId) {
    return { lastIntention: null, lastReflection: null, lastMode: null };
  }

  const { data: lastReflectionRow, error: reflectionError } = await supabase
    .from("reflections")
    .select("intention, mode, created_at")
    .eq("user_id", userId)
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (reflectionError) {
    console.error("[Spirit] Supabase reflections error:", reflectionError.message);
  }

  const { data: sessionRow, error: sessionError } = await supabase
    .from("sessions")
    .select("last_intention, last_mode")
    .eq("user_id", userId)
    .limit(1)
    .maybeSingle();

  if (sessionError) {
    console.error("[Spirit] Supabase sessions error:", sessionError.message);
  }

  return {
    lastIntention: sessionRow?.last_intention || null,
    lastReflection: lastReflectionRow?.intention || null,
    lastMode: sessionRow?.last_mode || lastReflectionRow?.mode || null,
  };
}

async function storeReflectionAndSession({ userId, prompt, mode, reply }) {
  if (!userId) return;

  const now = new Date().toISOString();

  const { error: insertError } = await supabase.from("reflections").insert({
    user_id: userId,
    intention: prompt,
    mode,
    content: JSON.stringify({ prompt, reply, mode }),
    created_at: now,
  });

  if (insertError) {
    console.error("[Spirit] Error inserting reflection:", insertError.message);
  }

  const { error: upsertError } = await supabase
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

  if (upsertError) {
    console.error("[Spirit] Error upserting session:", upsertError.message);
  }
}

// ─────────────────────────────────────────────
//  POST /chat — Main Intelligence Endpoint
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
      model: MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: text },
      ],
      temperature: TEMPERATURE,
      max_tokens: MAX_TOKENS,
    });

    const reply =
      completion.choices?.[0]?.message?.content?.trim() ||
      "I am here. What do you need right now?";

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
  userId: effectiveUserId || null,   // ← add this line
  ts: new Date().toISOString(),
});
  } catch (err) {
    console.error("[Spirit /chat error]", err?.message || err);

    return res.status(500).json({
      ok: false,
      error: "Spirit encountered an error while processing this request.",
    });
  }
});

export default router;
