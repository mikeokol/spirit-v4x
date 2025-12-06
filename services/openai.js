// services/openai.js
// Spirit v7 — OpenAI client wrapper (production clean)

import OpenAI from "openai";

// Important: This service is ONLY for calling OpenAI.
// No Supabase logic belongs in this file.

const apiKey = process.env.OPENAI_API_KEY;

if (!apiKey) {
  console.error("[OpenAI] Missing OPENAI_API_KEY — AI responses will fail.");
}

export const openai = new OpenAI({
  apiKey,
});

// Core wrapper for Spirit executor
export async function generateCompletion({ system, user, model = "gpt-4.1-mini" }) {
  try {
    const response = await openai.chat.completions.create({
      model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature: 0.7,
    });

    return {
      ok: true,
      text: response.choices?.[0]?.message?.content || "",
    };
  } catch (err) {
    console.error("[OpenAI Error]", err);
    return { ok: false, error: err?.message || "OpenAI failure" };
  }
}