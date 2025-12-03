// engine/executor.js — Spirit v7.2 Executor (Creator v2, JSON-Stable)
import { getClient } from "../services/openai.js";

function getResponseText(response) {
  if (response?.output?.[0]?.content?.[0]?.text) {
    return response.output[0].content[0].text;
  }
  if (response?.output_text) return response.output_text;
  return "";
}

export async function runExecutor({ userId, message, plan, memory, mode, taskType }) {
  const client = getClient();

  const safeMode = mode || "sanctuary";
  const safeTaskType = taskType || (safeMode === "creator" ? "creator_script" : "sanctuary_chat");
  const isCreator = safeMode === "creator";

  let roleBlock = "";

  if (isCreator) {
    roleBlock = `
You are **Spirit – Creator Engine v2**, behaving like a full media team in one brain.

You know how to:
- Select and adapt to **platforms** (TikTok, Reels, Shorts, YouTube, X, etc.).
- Clarify or infer the **niche & target audience** (who, what pain, what desire).
- Craft **elite hooks** that grab attention in the first 1–3 seconds.
- Design a **retention structure** (pattern breaks, open loops, payoffs, CTA).
- Generate **word-for-word scripts** when the user is stuck or says "I don't know what to say here".
- Suggest **hashtags** tuned to discoverability (not spammy).
- Suggest **thumbnail or visual ideas** (you may reference “use an AI thumbnail generator” in generic terms, but do not hard-promote specific brands).
- Give **delivery coaching**: tone, pacing, energy, body language, especially for camera-shy or low-confidence users.
- Use memory: if they’re shy, overwhelmed, or overthinking, adjust style: more reassuring, step-by-step, and specific.

You are working from this PLAN (steps) and MEMORY (what you know about this user so far).

You MUST produce a **single JSON object** with this EXACT shape:

{
  "reply": "What you want shown to the user in the app as Markdown. It should include clear sections (Platform, Hook, Script, Hashtags, Delivery Notes, Thumbnail Idea, Audience Strategy, Iteration Plan)."
}

Important:
- "reply" is a single string.
- Inside "reply", you can use Markdown headings and bullet points.
- Do NOT include any other top-level fields besides "reply".
`;
  } else {
    roleBlock = `
You are the **Executor** module of Spirit v7.2.

Your job:
- Take the PLAN and the user's message.
- Use MEMORY as context.
- Produce ONE unified, helpful reply as "reply" in a JSON object.

Return ONLY:

{
  "reply": "final message to the user"
}
`;
  }

  const prompt = `
${roleBlock}

UserId: ${userId}
Mode: ${safeMode}
TaskType: ${safeTaskType}

PLAN (from Planner):
${JSON.stringify(plan, null, 2)}

USER MESSAGE:
"""
${message}
"""

MEMORY SNAPSHOT (may be partial, do not invent facts):
${JSON.stringify(memory || {}, null, 2)}
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
    max_output_tokens: 900,
  });

  const text = getResponseText(response);

  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed.reply === "string") {
      return parsed.reply;
    }
  } catch (err) {
    console.warn("[executor] JSON parse failed, falling back to raw text:", err?.message || err);
  }

  // Fallback: never break the app, always give something usable
  if (text && typeof text === "string") {
    return text;
  }

  return "Executor fallback: unable to generate a structured response, but the system is online.";
}