// runTool.js — Trinity v7 Tool Execution Layer
// Executes tool calls detected by Executor

export async function runTool(call, tools, memory, context) {
  let parsed;

  try {
    parsed = typeof call === "string" ? JSON.parse(call) : call;
  } catch (err) {
    return {
      error: "INVALID_TOOL_CALL_JSON",
      raw: call,
      details: err.message
    };
  }

  if (!parsed.tool) {
    return {
      error: "MALFORMED_TOOL_CALL",
      parsed
    };
  }

  const { tool, input } = parsed;

  const toolFn = tools[tool];
  if (!toolFn) {
    return {
      error: "UNKNOWN_TOOL",
      tool,
      available: Object.keys(tools)
    };
  }

  try {
    // All Trinity v7 tools follow signature:
    // toolFn(input, memory, context)
    const result = await toolFn(input, memory, context);
    return { tool, result };
  } catch (err) {
    return {
      error: "TOOL_EXECUTION_FAILED",
      tool,
      details: err.stack || err.message
    };
  }
}