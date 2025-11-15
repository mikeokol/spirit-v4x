// routes/chat.js — Spirit v4.x Intelligence Layer (with chat-style memory + plan memory)
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
    return "creator";
  }

  // Hybrid explicit
  if (t.includes("hybrid system") || t.includes("mind•body•brand") || t.includes("mind body brand")) {
    return "hybrid";
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
    creator: "Focus on content, brand, storytelling, and creator leverage.",
    reflection:
      "Treat this as a reflection/intention log. Help the user name their state and give one clear next move.",
    oracle:
      "Zoom out to deeper questions of meaning, human nature, and perspective — but always end with a concrete action.",
    coach:
      "Act as a hybrid coach across mind, body, and brand, choosing the most relevant pillar for the request.",
    hybrid:
      "Act as a Mind•Body•Brand architect. Unify training, content, identity, and lifestyle.",
    sanctuary:
      "Act as the central sanctuary: respond as a hybrid identity guide across mind, body, and brand, with extra focus on presence and clarity.",
  }[mode] || "Act as a hybrid coach across mind, body, and brand.";

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

Core behavioral structure:
1. Identify the underlying desire or friction.
2. Clarify what matters most.
3. Prescribe 1–2 clear steps (only if needed).
4. Reinforce identity.

Tone:
${toneDescriptor}

Mode:
${mode.toUpperCase()}
${modeLine}

Previous context:
${previousContext}

Style rules:
- Replies: 3–7 sentences unless asked otherwise.
- You may use *one short action list* if it increases clarity.
- Avoid generic motivation.
- Presence first. Clarity second. Action third.

Your job: respond as Spirit v4.x with this identity, tone, and structure.
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
//  NEW — Plan Memory Helpers
// ─────────────────────────────────────────────
async function storePlan({ userId, type, mode, prompt, reply }) {
  if (!userId) return;
  try {
    const summary = (reply || "").split("\n").filter(Boolean)[0]?.slice(0, 240) || null;

    await supabase.from("plans").insert({
      user_id: userId,
      type,       // "fitness" | "creator" | "hybrid"
      mode,       // body | creator | hybrid
      summary,
      prompt,
      reply,
    });
  } catch (err) {
    console.error("[Spirit/storePlan] error:", err?.message || err);
  }
}

async function updateSessionPlanSummary({ userId, type, reply }) {
  if (!userId) return;
  try {
    const summary = (reply || "").split("\n").filter(Boolean)[0]?.slice(0, 240) || null;
    if (!summary) return;

    const now = new Date().toISOString();

    const fieldMap = {
      fitness: "last_fitness_plan_summary",
      creator: "last_creator_plan_summary",
      hybrid: "last_hybrid_plan_summary",
    };

    const fieldName = fieldMap[type];
    if (!fieldName) return;

    await supabase
      .from("sessions")
      .upsert(
        {
          user_id: userId,
          [fieldName]: summary,
          updated_at: now,
        },
        { onConflict: "user_id" }
      );
  } catch (err) {
    console.error("[Spirit/updateSessionPlanSummary] error:", err?.message || err);
  }
}

// ─────────────────────────────────────────────
//  POST /chat — main intelligence endpoint
// ─────────────────────────────────────────────
router.post("/", async (req, res) => {
  const { prompt, userId, sessionId, tone, mode: explicitMode, messages } = req.body || {};
  const text = typeof prompt === "string" ? prompt.trim() : "";

  if (!text) {
    return res.status(400).json({
      ok: false,
      error: "Missing 'prompt' in request body.",
    });
  }

  const effectiveUserId = userId || sessionId || null;

  try {
    const mode = explicitMode || classifyMode(text);

    const { lastIntention, lastReflection } = await getLastContext(effectiveUserId);

    const systemPrompt = buildSystemPrompt({
      mode,
      tone,
      lastIntention,
      lastReflection,
    });

    // Build chat-style history from frontend, if provided
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
        { role: "system", content: systemPrompt },
        ...historyMessages,
        { role: "user", content: text },
      ],
      temperature: Number(process.env.SPIRIT_TEMPERATURE || 0.6),
      max_tokens: Number(process.env.SPIRIT_MAX_TOKENS || 400),
    });

    const reply =
      completion.choices?.[0]?.message?.content?.trim() ||
      "I’m here. Let’s take one clear step. What do you need right now?";

    // Store reflections (existing logic)
    if (mode === "reflection" || text.toLowerCase().startsWith("i choose ")) {
      await storeReflectionAndSession({
        userId: effectiveUserId,
        prompt: text,
        mode,
        reply,
      });
    }

    // NEW — Plan Memory for Body/Creator/Hybrid modes
    const planModeMap = {
      body: "fitness",
      creator: "creator",
      hybrid: "hybrid",
    };

    const planType = planModeMap[mode];
    if (planType && effectiveUserId) {
      await storePlan({
        userId: effectiveUserId,
        type: planType,
        mode,
        prompt: text,
        reply,
      });

      await updateSessionPlanSummary({
        userId: effectiveUserId,
        type: planType,
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
