// engine/tools/runTool.js — Executes TOOL_CALL instructions from the Executor

export async function runTool(toolCall, tools) {
  let parsed;

  try {
    parsed = JSON.parse(toolCall);
  } catch (err) {
    return {
      error: "INVALID_TOOL_CALL_JSON",
      details: err.message,
      raw: toolCall,
    };
  }

  if (!parsed.TOOL_CALL) {
    return {
      error: "MALFORMED_TOOL_CALL",
      details: parsed,
    };
  }

  const { tool, input } = parsed.TOOL_CALL;
  const registry = tools || {};

  if (!registry[tool]) {
    return {
      error: "UNKNOWN_TOOL",
      tool,
      available: Object.keys(registry),
    };
  }

  try {
    return await registry[tool](input);
  } catch (err) {
    return {
      error: "TOOL_EXECUTION_FAILED",
      tool,
      details: err.message,
    };
  }
}