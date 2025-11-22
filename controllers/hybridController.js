// controllers/hybridController.js — Spirit v5.1 Showcase Mode

import { buildFitnessBlock } from "../utils/fitnessEngine.js";
import { buildCreatorContent } from "../utils/creatorEngine.js";
import { spiritVoice } from "../utils/spiritVoice.js";

export const hybridController = async (req, res) => {
  try {
    const {
      goal,
      experience,
      days,
      topic,
      platform,
      identity,
      tone
    } = req.body;

    if (!goal || !experience || !days || !topic) {
      return res.status(400).json({
        ok: false,
        error: "Missing required fields: goal, experience, days, topic"
      });
    }

    // Build adaptive fitness block
    const fitness = buildFitnessBlock({ goal, experience, days });

    // Build structured creator content
    const content = buildCreatorContent({ topic, platform, identity, tone });

    // Unified Spirit persona narration
    const narration = spiritVoice(
      "hybrid",
      `Goal: ${goal}, Topic: ${topic}`
    );

    return res.json({
      ok: true,
      mode: "hybrid",
      fitness,
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
