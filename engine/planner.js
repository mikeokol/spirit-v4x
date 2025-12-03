// engine/planner.js — Spirit v7.2 Planner (Creator-Aware, JSON-Stable)
import { getClient } from "../services/openai.js";

function getResponseText(response) {
  // New Responses API shape
  if (response?.output?.[0]?.content?.[0]?.text) {
    return response.output[0].content[0].text;
  }
  // Older helper / convenience field
  if (response?.output_text) return response.output_text;
  // Last resort
  return "";
}

export async function runPlanner({ userId, message, mode, taskType, memory }) {
  const client = getClient();

  const safeMode = mode || "sanctuary";
  const safeTaskType = taskType || (safeMode === "creator" ? "creator_script" : "sanctuary_chat");

  const isCreator = safeMode === "creator";

  const creatorBlock = isCreator
    ? `
You are the **Creator Mode Planner v2** for Spirit.

Your job is to design a structured plan that feels like a *full media team* supporting the user.

ALWAYS think in terms of:
- Platform context (e.g. TikTok, YouTube Shorts, Reels, X, etc.)
- Audience + niche (who they’re talking to, what they care about)
- Hook + angle (how to grab attention in first 1–3 seconds)
- Structure for retention (pattern breaks, open loops, payoffs)
- Script depth (word-for-word option if user is stuck)
- Distribution (hashtags, posting pattern, cross-posting)
- Iteration (how to review performance and improve next posts)
- Personalization (use any memory you have about confidence, camera shyness, goals, etc.)

Design **2–6 clear steps** that an EXECUTOR can follow to produce a full “content pack” for the user (script + strategy, not just words.
`
    : `
You are the **Planner** module of Spirit v7.2.
Your job is to break the user’s request into a small, crisp, actionable plan (2–6 steps)
that the EXECUTOR can follow to produce a strong reply.
`;

  const prompt = `
${creatorBlock}

UserId: ${userId}
Mode: ${safeMode}
TaskType: ${safeTaskType}

User message:
"""
${message}
"""

Current memory snapshot (may be empty but do NOT invent facts):
${JSON.stringify(memory || {}, null, 2)}

Return **ONLY valid JSON** in this exact format:

{
  "plan": {
    "taskType": "${safeTaskType}",
    "mode": "${safeMode}",
    "steps": [
      { "id": "step-1", "action": "short_snake_case_action", "detail": "clear human-readable description" }
    ],
    "checks": [
      "Is output aligned with taskType?",
      "Is persona consistent?",
      "Is final answer unified and actionable?"
    ]
  }
}
`;

  const response = await client.responses.create({
    model: "gpt-4.1-mini",
    input: [
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: prompt,
          },
        ],
      },
    ],
    max_output_tokens: 500,
  });

  const text = getResponseText(response);

  try {
    const parsed = JSON.parse(text);
    if (parsed && parsed.plan) {
      return parsed.plan;
    }
  } catch (_) {
    // fall through
  }

  // Fallback – never break the engine
  return {
    taskType: safeTaskType,
    mode: safeMode,
    steps: [
      {
        id: "fallback",
        action: "fallback_plan",
        detail: "Planner failed to produce valid JSON; EXECUTOR should still generate a helpful, safe reply.",
      },
    ],
    checks: [
      "Is output aligned with taskType?",
      "Is persona consistent?",
      "Is final answer unified and actionable?",
    ],
  };
}