import { openai } from "../services/openai.js";

export async function analyzeLiveMessage({ message, state }) {
  const prompt = `
You are Spirit v7, an elite live fitness coach.
Classify the user's message into one of these categories:

- "action" (done, ready, continue, next)
- "tip" (asking form guidance or technique)
- "fatigue" (tired, hard, struggling)
- "pain" (injury or pain warning)
- "rest" (asking for break)
- "motivation" (asking for push or encouragement)
- "progress" (asking how many sets left etc)
- "question" (general)
- "other"

User message: "${message}"
Current phase: ${state.phase}
Current block: ${state.block}
`;

  const result = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "system", content: prompt }],
  });

  const type = result.choices[0].message.content.trim();
  return type.toLowerCase();
}

// Context-aware coaching responses
export async function generateLiveCoachResponse({ type, message, state }) {
  const basePrompt = `
You are Spirit v7’s elite live fitness coach.
Give short, precise responses.
Maintain user safety.
Never break the workout flow.

State:
- Phase: ${state.phase}
- Block: ${state.block}
- Set: ${state.set || 1}
`;

  let instruction = "";

  switch (type) {
    case "tip":
      instruction = `Give one expert tip for the current exercise.`;
      break;

    case "fatigue":
      instruction = `Give a supportive response and reduce intensity slightly.`;
      break;

    case "pain":
      instruction = `Tell them to stop immediately and modify/switch exercise.`;
      break;

    case "rest":
      instruction = `Approve a short rest and tell them when to resume.`;
      break;

    case "motivation":
      instruction = `Give an elite coach motivational line.`;
      break;

    case "progress":
      instruction = `Tell how many sets remain in this block.`;
      break;

    case "question":
      instruction = `Answer clearly and keep the workout flow.`;
      break;

    default:
      instruction = `Give a neutral acknowledgement and keep the workout flow.`;
  }

  const result = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: basePrompt },
      { role: "user", content: message },
      { role: "system", content: instruction },
    ],
  });

  return result.choices[0].message.content.trim();
}