// engine/executor.js — Spirit v7 Cognitive Engine Executor

import fs from "fs";
import { openai } from "../services/openai.js";
import { runTool } from "./tools/runTool.js";

// Load system prompt for executor
const executorSystemPrompt = fs.readFileSync("./prompts/executor.txt", "utf8");

export async function spiritExecutor(plan, memory, tools) {
  const safePlan = plan || {};
  const steps = Array.isArray(safePlan.steps) ? safePlan.steps : [];
  const safeTools = tools || {};
  const results = [];

  for (const step of steps) {
    const stepPayload = {
      step,
      memory: memory || {},
      toolsAvailable: Object.keys(safeTools),
    };

    const result = await openai.chat.completions.create({
      model: "gpt-5.1",
      messages: [
        { role: "system", content: executorSystemPrompt },
        {
          role: "user",
          content: JSON.stringify(stepPayload, null, 2),
        },
      ],
    });

    const output = result.choices?.[0]?.message?.content ?? "";

    // Tool call convention: model returns a JSON string containing TOOL_CALL
    if (output.includes("TOOL_CALL")) {
      const toolResult = await runTool(output, safeTools);
      results.push(toolResult);
      continue;
    }

    // Otherwise it's a normal text result
    results.push(output);
  }

  return results.join("\n\n");
}