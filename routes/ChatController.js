// routes/ChatController.js
// ---------------------------------------------------------------------------
// Spirit v5.1 — Main Intelligence Controller
// Orchestrates all engines (prompt, creator, fitness, context)
// ---------------------------------------------------------------------------

import OpenAI from "openai";
import { classifyMessage } from "./classifier.js";

import {
  buildSystemPrompt,
  FITNESS_PLAN_RUBRIC,
} from "./PromptEngine.js";

import {
  detectFitnessPrompt,
  parseFitnessMeta,
  buildFitnessPrompt,
  extractWorkouts,
} from "./FitnessEngine.js";

import {
  getLastContext,
  storeReflection,
  storeTrainingBlock,
} from "./ContextEngine.js";

import { creatorSystemPrompt } from "./CreatorEngine.js";

// ---------------------------------------------------------------------------
// Main handler
// ---------------------------------------------------------------------------
export async function handleChat(req, res) {
  const {
    prompt,
    userId,
    sessionId,
    messages,
    goalCategory,
    specificGoal,
    experience,
    days,
    gender,
    weight,
    height,
  } = req.body || {};

  if (!prompt || typeof prompt !== "string") {
    return res.status(400).json({ ok: false, error: "Missing prompt" });
  }

  const text = prompt.trim();
  const effectiveUserId = userId || sessionId || null;

  try {
    // ----------------------------------------
    // STEP 1 — classify mode
    // ----------------------------------------
    const mode = await classifyMessage(text);

    const client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    const model = process.env.SPIRIT_MODEL || "gpt-4o-mini";

    // ----------------------------------------
    // STEP 2 — Creator Mode (special branch)
    // ----------------------------------------
    if (mode === "creator") {
      const safeHistory = Array.isArray(messages)
        ? messages
            .filter(
              (m) =>
                m &&
                typeof m.content === "string" &&
                (m.role === "user" || m.role === "assistant")
            )
            .slice(-8)
        : [];

      const completion = await client.chat.completions.create({
        model,
        messages: [
          { role: "system", content: creatorSystemPrompt },
          ...safeHistory,
          { role: "user", content: text },
        ],
        temperature: Number(process.env.SPIRIT_TEMPERATURE || 0.7),
        max_tokens: Number(process.env.SPIRIT_MAX_TOKENS || 800),
      });

      const reply =
        completion.choices?.[0]?.message?.content?.trim() ||
        "Here are some content ideas.";

      return res.json({
        ok: true,
        service: "Spirit v5.1",
        mode,
        reply,
        ts: new Date().toISOString(),
      });
    }

    // ----------------------------------------
    // STEP 3 — Load previous context (reflection + session)
    // ----------------------------------------
    const { lastIntention, lastReflection } =
      await getLastContext(effectiveUserId);

    // ----------------------------------------
    // STEP 4 — Build system prompt
    // ----------------------------------------
    let systemPrompt = buildSystemPrompt({
      mode,
      lastIntention,
      lastReflection,
    });

    let userPrompt = text;
    let isFitnessPlan = false;
    let fitnessMeta = null;

    // ----------------------------------------
    // STEP 5 — Fitness branch
    // ----------------------------------------
    if (
      mode === "body" &&
      (detectFitnessPrompt(text) ||
        goalCategory ||
        experience ||
        days ||
        weight ||
        height)
    ) {
      isFitnessPlan = true;

      const parsedMeta = parseFitnessMeta(text);

      fitnessMeta = {
        ...parsedMeta,
        goal: goalCategory || parsedMeta.goal,
        specificGoal: specificGoal || parsedMeta.specificGoal,
        experience: experience || parsedMeta.experience,
        days: days || parsedMeta.days,
        gender: gender || parsedMeta.gender,
        weight: weight || parsedMeta.weight,
        height: height || parsedMeta.height,
        tone: parsedMeta.tone || "default",
      };

      userPrompt = buildFitnessPrompt(fitnessMeta);

      systemPrompt += "\n\n" + FITNESS_PLAN_RUBRIC;
    }

    // ----------------------------------------
    // STEP 6 — Prepare short history for chat continuity
    // ----------------------------------------
    const safeHistory = Array.isArray(messages)
      ? messages
          .filter(
            (m) =>
              m &&
              typeof m.content === "string" &&
              (m.role === "user" || m.role === "assistant")
          )
          .slice(-8)
      : [];

    // ----------------------------------------
    // STEP 7 — Generate core Spirit reply
    // ----------------------------------------
    const completion = await client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: systemPrompt },
        ...safeHistory,
        { role: "user", content: userPrompt },
      ],
      temperature: Number(process.env.SPIRIT_TEMPERATURE || 0.6),
      max_tokens: Number(process.env.SPIRIT_MAX_TOKENS || 800),
    });

    const reply =
      completion.choices?.[0]?.message?.content?.trim() ||
      "I’m here. What direction would you like to take next?";

    // ----------------------------------------
    // STEP 8 — Store reflection if needed
    // ----------------------------------------
    if (mode === "reflection" || text.toLowerCase().startsWith("i choose ")) {
      await storeReflection({
        userId: effectiveUserId,
        prompt: text,
        mode,
        reply,
      });
    }

    // ----------------------------------------
    // STEP 9 — Store training block (fitness)
    // ----------------------------------------
    if (isFitnessPlan && effectiveUserId) {
      const workouts = await extractWorkouts(
        reply,
        model,
        process.env.OPENAI_API_KEY
      );

      await storeTrainingBlock({
        userId: effectiveUserId,
        planText: reply,
        meta: fitnessMeta,
        workouts,
      });
    }

    // ----------------------------------------
    // STEP 10 — Final response
    // ----------------------------------------
    return res.json({
      ok: true,
      service: "Spirit v5.1",
      mode,
      reply,
      ts: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[Spirit ChatController Error]", err.message);

    return res.status(500).json({
      ok: false,
      error: "Spirit encountered an error.",
      details: err.message,
    });
  }
}