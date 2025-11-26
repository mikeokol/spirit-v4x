// planner.js — Trinity v7 Cognitive Engine Planner
// Purpose: Convert taskType + prompt into a structured plan for executor.

import { openai } from "../services/openai.js";

export async function spiritPlanner(taskType, prompt, memory, context = {}) {
  const planningQuery = `
You are the PLANNER of the Trinity v7 Cognitive Engine.
Your job: break the task into a short JSON plan.

Follow this schema:
{
  "taskType": "...",
  "steps": [
    { "id": "step-1", "action": "explain", "detail": "..." },
    { "id": "step-2", "action": "process", "detail": "..." },
    { "id": "step-3", "action": "finalize", "detail": "..." }
  ],
  "checks": [
    "Is output aligned with taskType?",
    "Is memory context considered?",
    "Is final output coherent and safe?"
  ]
}

Task Type: ${taskType}
Prompt: ${prompt}
Memory Summary: ${JSON.stringify(memory)}
Context: ${JSON.stringify(context)}

ONLY RETURN VALID JSON. NOTHING ELSE.
`;

  const response = await openai.chat.completions.create({
    model: "gpt-4.1-mini",
    messages: [
      { role: "system", content: "You are the structured planner of Trinity v7." },
      { role: "user", content: planningQuery }
    ]
  });

  // Extract raw text
  const raw = response.choices[0].message.content.trim();

  // Fail-safe parsing
  try {
    return JSON.parse(raw);
  } catch (err) {
    console.error("Planner JSON error:", raw);
    return {
      taskType,
      steps: [{ id: "fallback", action: "respond", detail: prompt }],
      checks: ["Failed to parse plan — using fallback."]
    };
  }
}