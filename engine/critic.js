// critic.js — Trinity v7 Cognitive Engine Critic
// Purpose: validate executor output with structured JSON feedback

import { openai } from "../services/openai.js";

export async function spiritCritic(output, plan, memory) {
  const query = `
You are the CRITIC module of the Trinity v7 Cognitive Engine.

Your job is to evaluate whether the final output fully satisfies the plan,
and whether it is coherent, safe, on-topic, and aligned with the task type.

Return strictly in JSON:

{
  "ok": true | false,
  "notes": "explanation if revision needed"
}

Evaluation Context:
- Task Type: ${plan.taskType}
- Steps: ${JSON.stringify(plan.steps)}
- Output: ${JSON.stringify(output)}
- Memory Snapshot: ${JSON.stringify(memory)}
`;

  const response = await openai.chat.completions.create({
    model: "gpt-4.1-mini",
    messages: [
      { role: "system", content: "You are the strict critic of Trinity v7." },
      { role: "user", content: query }
    ]
  });

  const raw = response.choices[0].message.content.trim();

  // Parse safely
  try {
    const parsed = JSON.parse(raw);
    return parsed;
  } catch (err) {
    console.error("Critic JSON parse error:", raw);
    return {
      ok: true,
      notes: "Critic failed to parse JSON; allowing output."
    };
  }
}