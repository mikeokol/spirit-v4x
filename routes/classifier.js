// routes/classifier.js
// ---------------------------------------------------------------------------
// Spirit v5.1 — Mode Classifier
// Determines the conversational mode: reflection, mind, body, brand, creator,
// oracle, hybrid, coach, sanctuary.
// First: rule-based (fast, reliable).
// Optional: AI-assisted passthrough for edge cases.
// ---------------------------------------------------------------------------

import OpenAI from "openai";

// ---------------------------------------------------------------------------
// CONFIG
// ---------------------------------------------------------------------------
const ENABLE_AI_FALLBACK = false; // Set to true if you want deeper classification
const MODEL = process.env.SPIRIT_CLASSIFIER_MODEL || "gpt-4o-mini";

// ---------------------------------------------------------------------------
// RULE-BASED DETECTION (Primary, most reliable)
// ---------------------------------------------------------------------------
export async function classifyMessage(text = "") {
  if (!text || typeof text !== "string") return "coach";

  const t = text.toLowerCase().trim();

  // -------------------------
  // Reflection mode
  // -------------------------
  if (
    t.startsWith("i choose ") ||
    t.includes("my intention is") ||
    t.includes("i want to reflect") ||
    t.includes("life reflection")
  ) {
    return "reflection";
  }

  // -------------------------
  // Body / Fitness mode
  // -------------------------
  const bodyKeywords = [
    "workout",
    "training",
    "hypertrophy",
    "strength",
    "calories",
    "diet",
    "nutrition",
    "fitness",
    "bulk",
    "cut",
    "lose weight",
    "gain muscle",
    "training days",
    "experience level",
    "build a body coaching",
    "training block",
  ];
  if (bodyKeywords.some((k) => t.includes(k))) return "body";

  // -------------------------
  // Creator mode (Content Engine)
  // -------------------------
  const creatorKeywords = [
    "youtube",
    "tiktok",
    "content ideas",
    "content plan",
    "posting schedule",
    "hashtags",
    "script",
    "thumbnails",
    "viral",
    "video ideas",
    "creator",
  ];
  if (creatorKeywords.some((k) => t.includes(k))) return "creator";

  // -------------------------
  // Brand (identity + storytelling)
  // -------------------------
  if (
    t.includes("brand") ||
    t.includes("storytelling") ||
    t.includes("audience") ||
    t.includes("narrative")
  ) {
    return "brand";
  }

  // -------------------------
  // Oracle (philosophy / life meaning / clarity)
  // -------------------------
  const oracleKeywords = [
    "purpose",
    "universe",
    "meaning",
    "consciousness",
    "philosophy",
    "wisdom",
    "big picture",
  ];
  if (oracleKeywords.some((k) => t.includes(k))) return "oracle";

  // -------------------------
  // Hybrid (mind+body+brand blended)
  // -------------------------
  if (t.includes("hybrid") || t.includes("mind and body") || t.includes("integration")) {
    return "hybrid";
  }

  // -------------------------
  // Sanctuary (identity grounding)
  // -------------------------
  if (
    t.includes("sanctuary") ||
    t.includes("ground me") ||
    t.includes("help me breathe") ||
    t.includes("identity reset")
  ) {
    return "sanctuary";
  }

  // -------------------------
  // Mind (default high-level mode)
  // -------------------------
  const mindKeywords = ["identity", "discipline", "clarity", "mindset", "focus"];
  if (mindKeywords.some((k) => t.includes(k))) return "mind";

  // -------------------------
  // Coach (fallback; generic guidance)
  // -------------------------
  if (!ENABLE_AI_FALLBACK) return "coach";

  // -------------------------
  // OPTIONAL — AI FALLBACK (only for edge cases)
  // -------------------------
  try {
    const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

    const completion = await client.chat.completions.create({
      model: MODEL,
      max_tokens: 50,
      temperature: 0.0,
      messages: [
        {
          role: "system",
          content: `
Classify the user's intent into ONE mode:
"reflection", "mind", "body", "brand", "creator", "oracle", "hybrid", "sanctuary", or "coach".

Respond with ONLY the label, nothing else.
`,
        },
        { role: "user", content: text },
      ],
    });

    const answer = completion.choices?.[0]?.message?.content?.trim().toLowerCase();
    const validModes = [
      "reflection",
      "mind",
      "body",
      "brand",
      "creator",
      "oracle",
      "hybrid",
      "sanctuary",
      "coach",
    ];

    if (validModes.includes(answer)) return answer;
    return "coach";
  } catch (err) {
    console.warn("[classifier AI fallback error]", err.message);
    return "coach";
  }
}