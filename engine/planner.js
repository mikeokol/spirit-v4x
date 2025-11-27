// engine/planner.js — Spirit v7.1 Planner (chat.completions only)

import { getClient } from "../services/openai.js";

/**
 * Extracts the first JSON object from a string, stripping ``` fences if present.
 */
function extractJsonObject(text) {
  if (!text) return null;

  let cleaned = text.trim();

  // Strip markdown fences if any
  if (cleaned.startsWith("```")) {
    cleaned = cleaned.replace(/^```[a-zA-Z]*\s*/, "").replace(/```$/, "").trim();
  }

  const firstBrace = cleaned.indexOf("{");
  const lastBrace = cleaned.lastIndexOf("}");
  if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
    return null;
  }

  const candidate = cleaned.slice(firstBrace, lastBrace + 1);
  try {
    return JSON.parse(candidate);
  } catch {
    return null;
  }
}

/**
 * Planner: turns (message + mode + taskType + memory) into a structured plan.
 * If JSON fails, returns a safe fallback plan.
 */
export async function runPlanner({ userId, message, mode, taskType, memory }) {
  const client = getClient();

  const userPrompt = `
You are the PLANNER module of **Spirit v7.1**.

Your job:
- Turn the user message into a small, structured plan of 2–5 steps.
- Steps should be concrete actions the EXECUTOR can follow.
- You MUST answer in **pure JSON only** (no prose, no explanation).

Context:
- userId: ${userId}
- mode: ${mode}
- taskType: ${taskType}

User message:
${message}

Memory snapshot (may be partial):
${JSON.stringify(memory ?? {}, null, 2)}

Return JSON with EXACTLY this shape:

{
  "plan": {
    "taskType": "<repeat taskType or infer>",
    "mode": "<repeat mode or infer>",
    "steps": [
      { "id": "step-1", "action": "short_snake_case_action", "detail": "Clear description of what to do." }
    ],
    "checks": [
      "Is output aligned with taskType?",
      "Is persona consistent?",
      "Is final answer unified and actionable?"
    ]
  }
}

NO extra top-level fields, NO comments, NO markdown, NO backticks.
`;

  const completion = await client.chat.completions.create({
    model: "gpt-4.1-mini",
    messages: [
      {
        role: "system",
        content:
          "You are the Spirit v7.1 Planner. You ONLY return strict JSON matching the requested schema. No explanations.",
      },
      {
        role: "user",
        content: userPrompt,
      },
    ],
    temperature: 0.2,
  });

  const raw = completion.choices?.[0]?.message?.content ?? "";
  const parsed = extractJsonObject(raw);

  if (parsed && parsed.plan) {
    return parsed.plan;
  }

  // Fallback: never crash the engine
  return {
    taskType: taskType || "generic",
    mode: mode || "sanctuary",
    steps: [
      {
        id: "fallback",
        action: "fallback_plan",
        detail:
          "Planner could not parse valid JSON. Executor should still generate a helpful, unified answer directly from the user message and memory.",
      },
    ],
    checks: [
      "Is output aligned with taskType?",
      "Is persona consistent?",
      "Is final answer unified and actionable?",
    ],
  };
}