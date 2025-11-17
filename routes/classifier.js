// routes/classifier.js — Spirit v5.0 LLM Mode Classifier
// --------------------------------------------------------
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Cached classifications to reduce cost
const cache = new Map();

// Possible modes Spirit understands
const MODES = [
  "reflection",
  "mind",
  "body",
  "brand",
  "creator",
  "oracle",
  "coach",
  "hybrid",
  "sanctuary"
];

// Prompt that guides classification
const CLASSIFIER_SYSTEM_PROMPT = `
You are a strict classifier for an AI called Spirit v5.0.

Your job: Read the user's message and classify it into ONE AND ONLY ONE mode.
Return ONLY the mode name, lowercase, no punctuation, no explanation.

Valid modes are:
reflection — "I choose ..." statements, journaling, intentions, self-truth.
mind — mental clarity, discipline, consistency, identity.
body — workout, diet, physical goals, training blocks, fitness.
brand — content ideas, scripts, thumbnails, messaging, marketing.
creator — large-scale creative systems: content engines, posting strategy.
oracle — philosophy, meaning, purpose, human nature, universe questions.
coach — general guidance that does not fit any specialized category.
hybrid — explicit blend of mind/body/brand in one question.
sanctuary — deeply personal or grounding messages seeking identity alignment.

Rules:
- If user mentions building a training plan → body.
- If user mentions content systems → creator.
- If message fits more than one category → return "hybrid".
- Do NOT explain. Only return the single mode word.
`.trim();

export async function classifyMessage(text = "") {
  const cleaned = text.toLowerCase().trim();

  if (cache.has(cleaned)) return cache.get(cleaned);

  const completion = await client.chat.completions.create({
    model: process.env.SPIRIT_CLASSIFIER_MODEL || "gpt-4o-mini",
    messages: [
      { role: "system", content: CLASSIFIER_SYSTEM_PROMPT },
      { role: "user", content: cleaned }
    ],
    temperature: 0,
    max_tokens: 5,
  });

  let mode = completion.choices?.[0]?.message?.content?.trim().toLowerCase();

  if (!MODES.includes(mode)) mode = "coach";

  cache.set(cleaned, mode);
  return mode;
}
