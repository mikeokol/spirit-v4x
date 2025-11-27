// engine/critic.js — Spirit v7 Cognitive Engine Critic

import fs from "fs";
import { openai } from "../services/openai.js";

// Load critic system prompt (tone, coherence, safety)
const criticSystemPrompt = fs.readFileSync("./prompts/critic.txt", "utf8");

export async function spiritCritic(finalDraft, userContext) {
  const response = await openai.chat.completions.create({
    model: "gpt-5.1",
    messages: [
      { role: "system", content: criticSystemPrompt },
      {
        role: "user",
        content: JSON.stringify(
          {
            draft: finalDraft ?? "",
            context: userContext ?? null,
          },
          null,
          2
        ),
      },
    ],
  });

  const verdict = response.choices?.[0]?.message?.content ?? "";

  if (verdict.includes("REVISION_REQUIRED")) {
    return { ok: false, notes: verdict };
  }

  return { ok: true, notes: verdict };
}