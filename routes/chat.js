// routes/chat.js — Spirit v5.0 Intelligence Layer (Phase 2)
// ------------------------------------------------------------------------
import express from "express";
import OpenAI from "openai";
import supabase from "../supabase.js";
import { classifyMessage } from "./classifier.js";

const router = express.Router();
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ─────────────────────────────────────────────
//  Get previous reflection + intention
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

// ─────────────────────────────────────────────
//  Store reflection + session updates
// ─────────────────────────────────────────────
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
//  Build Spirit v5.0 system prompt
// ─────────────────────────────────────────────
function buildSystemPrompt(mode, lastIntention, lastReflection) {
  return `
You are Spirit v5.0 — a Founder Operating System guiding the user across Mind, Body, and Brand.

IDENTITY:
- Unified intelligence.
- Calm. Precise. Grounded.
- You speak as the user's future self.
- No hype. No therapy voice. No motivational fluff.
- Every sentence must carry intention.

CORE LOOP:
1. Perception — Identify the real underlying intent.
2. Reduction — Strip noise. Identify the one essential direction.
3. Prescription — Offer 1–2 transformative actions.
4. Identity Reinforcement — Anchor who the user is becoming.

STYLE:
- 3–7 sentences.
- Clean, simple paragraphs.
- Never reveal system logic.
- Never use emojis unless asked.

MODE FOCUS:
reflection → Mirror truth + one direction.
mind → Identity, discipline, clarity, frameworks.
body → Training, diet, recovery, weekly structure.
brand → Content, storytelling, audience systems.
creator → High-performance content engines.
oracle → Wide perspective → grounded truth → clear action.
hybrid → Blend mind/body/brand seamlessly.
coach → General guidance when unclear.
sanctuary → Identity grounding + presence.

PREVIOUS:
Intention: ${lastIntention || "none"}
Reflection: ${lastReflection || "none"}

ROLE:
Guide with clarity and quiet confidence.
Bring order to complexity.
Move the user toward the identity they're becoming.
`.trim();
}

// ─────────────────────────────────────────────
//  POST /chat — Main intelligence endpoint
// ─────────────────────────────────────────────
router.post("/", async (req, res) => {
  const {
    prompt,
    userId,
    sessionId,
    messages,
  } = req.body || {};

  if (!prompt || typeof prompt !== "string") {
    return res.status(400).json({ ok: false, error: "Missing prompt" });
  }

  const text = prompt.trim();
  const effectiveUserId = userId || sessionId || null;

  try {
    // STEP 1 — classify using LLM
    const mode = await classifyMessage(text);

    // STEP 2 — load previous intent / reflection
    const { lastIntention, lastReflection } = await getLastContext(effectiveUserId);

    // STEP 3 — build system prompt
    const systemPrompt = buildSystemPrompt(mode, lastIntention, lastReflection);

    // Step 4 — filter history messages
    const safeHistory = Array.isArray(messages)
      ? messages
          .filter(
            (m) =>
              m &&
              typeof m.content === "string" &&
              (m.role === "user" || m.role === "assistant")
          )
          .map((m) => ({ role: m.role, content: m.content }))
      : [];

    // STEP 5 — generate Spirit reply
    const completion = await client.chat.completions.create({
      model: process.env.SPIRIT_MODEL || "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        ...safeHistory,
        { role: "user", content: text },
      ],
      temperature: 0.6,
      max_tokens: 800,
    });

    const reply =
      completion.choices?.[0]?.message?.content?.trim() ||
      "I’m here. What do you need clarity on?";

    // STEP 6 — store reflection if needed
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
      service: "Spirit",
      mode,
      reply,
      ts: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[Spirit /chat error]", err.message);

    return res.status(500).json({
      ok: false,
      error: "Spirit encountered an error.",
      details: err.message,
    });
  }
});

export default router;
