// executor.js — Trinity v7 Cognitive Engine Executor
// Purpose: execute planner steps into coherent final output.

import { openai } from "../services/openai.js";

export async function spiritExecutor(plan, execContext) {
  const { prompt, memory, tools, tone, context } = execContext;

  const results = [];

  for (const step of plan.steps) {
    const query = `
You are the EXECUTOR of the Trinity v7 Cognitive Engine.

Task Type: ${plan.taskType}
Step ID: ${step.id}
Action: ${step.action}
Detail: ${step.detail}

Prompt: ${prompt}
Tone: ${tone}
Context: ${JSON.stringify(context)}
Memory Summary: ${JSON.stringify(memory)}

If this step requires a tool, respond ONLY with:
{
  "tool": "<toolName>",
  "input": { ... }
}

Otherwise, respond ONLY with:
{
  "text": "..."
}
`;

    const response = await openai.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        { role: "system", content: "You are the executor of Trinity v7." },
        { role: "user", content: query }
      ]
    });

    let raw = response.choices[0].message.content.trim();

    // Fail-safe parse
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      console.error("Executor JSON error:", raw);
      parsed = { text: raw };
    }

    // If model triggered a tool
    if (parsed.tool) {
      const toolFunc = tools[parsed.tool];
      if (toolFunc) {
        const result = await toolFunc(parsed.input, memory, context);
        results.push(result);
        continue;
      } else {
        results.push(`[ERROR: Unknown tool '${parsed.tool}']`);
        continue;
      }
    }

    // Plain text result
    if (parsed.text) {
      results.push(parsed.text);
      continue;
    }

    // Fallback
    results.push(raw);
  }

  // Join all step outputs
  return results.join("\n\n");
}