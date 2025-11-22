// controllers/creatorController.js — Spirit v5.1 Showcase Creator Engine

import { spiritVoice } from "../utils/spiritVoice.js";
import { buildCreatorContent } from "../utils/creatorEngine.js";

export const creatorController = async (req, res) => {
  try {
    const { topic, platform, identity, tone } = req.body;

    if (!topic) {
      return res.status(400).json({
        ok: false,
        error: "Missing required field: topic"
      });
    }

    // Build structured content
    const content = buildCreatorContent({
      topic,
      platform: platform || "tiktok",
      identity,
      tone
    });

    // Apply Spirit persona voice
    const narration = spiritVoice("creator", `Topic: ${topic}`);

    return res.json({
      ok: true,
      mode: "creator",
      content,
      narration
    });
  } catch (err) {
    res.status(500).json({
      ok: false,
      error: err.message
    });
  }
};
