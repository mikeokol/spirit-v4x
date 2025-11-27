// engine/executor.js — Spirit v7.1 Executor (chat.completions, plain text)

import { getClient } from "../services/openai.js";

function modePersona(mode) {
  switch (mode) {
    case "fitness":
      return "You are Spirit in FITNESS mode: elite, direct, but supportive. You design practical, safe, progressive plans.";
    case "creator":
      return "You are Spirit in CREATOR mode: sharp, structured, focused on hooks, story, and clarity.";
    case "reflection":
      return "You are Spirit in REFLECTION mode: calm, probing, insight-driven. You ask good questions and help the user see patterns.";
    case "hybrid":
      return "You are Spirit in HYBRID mode: you blend training, mindset, and creator output into one coherent plan.";
    case "live":
      return "You are Spirit in LIVE coaching mode (Elite Founder): high-intensity but grounded, pushing the user toward concrete action.";
    case "sanctuary":
    default:
      return "You are Spirit in SANCTUARY mode: calm, precise, protective, and honest. You give clear next steps without fluff.";
  }
}

/**
 * Executor: follows the plan + memory and returns ONE unified text reply.
 * No JSON out, so it cannot break from parsing.
 */
export async function runExecutor({
  userId,
  message,
  plan,
  memory,
  mode,
  taskType,
}) {
  const client = getClient();

  const systemContent = `
You are the EXECUTOR module of Spirit v7.1.

${modePersona(mode)}

Rules:
- Follow the provided plan steps as guidance, but you may compress or adapt them.
- Output **ONE** unified answer as plain text (no JSON, no markdown fences required).
- Keep it **practical, specific, and grounded**.
- Respect the taskType: ${taskType}.
- If the plan has id "fallback", ignore its structure and respond directly to the user in the best possible way.
`;

  const userContent = `
UserId: ${userId}
Mode: ${mode}
TaskType: ${taskType}

User message:
${message}

Plan object:
${JSON.stringify(plan ?? {}, null, 2)}

Memory snapshot:
${JSON.stringify(memory ?? {}, null, 2)}
`;

  const completion = await client.chat.completions.create({
    model: "gpt-4.1-mini",
    messages: [
      { role: "system", content: systemContent },
      { role: "user", content: userContent },
    ],
    temperature: 0.45,
  });

  const reply = completion.choices?.[0]?.message?.content?.trim();
  if (!reply) {
    return "Executor fallback: model returned no content. Try again or adjust your request.";
  }

  return reply;
}