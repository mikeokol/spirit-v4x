// engine/planner.js — Spirit v7 Cognitive Engine Planner

import fs from "fs";
import { openai } from "../services/openai.js";

// Load planner system prompt
const plannerSystemPrompt = fs.readFileSync("./prompts/planner.txt", "utf8");

export async function spiritPlanner(taskType, payload) {
  const prompt = `
Task Type: ${taskType}
Payload: ${JSON.stringify(payload)}
  `.trim();

  const response = await openai.chat.completions.create({
    model: "gpt-5.1",
    messages: [
      { role: "system", content: plannerSystemPrompt },
      { role: "user", content: prompt },
    ],
  });

  const raw = response.choices?.[0]?.message?.content ?? "{}";

  try {
    return JSON.parse(raw);
  } catch (err) {
    console.error("Planner JSON parse error:", err, raw);
    // Safe fallback plan
    return {
      taskType,
      steps: [
        {
          id: "fallback",
          action: "respond",
          detail: "Generate a direct, helpful response to the user.",
        },
      ],
      checks: [],
    };
  }
}