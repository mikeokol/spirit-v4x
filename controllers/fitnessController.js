// controllers/fitnessController.js — Spirit v5.1 Showcase Fitness Engine
import { spiritVoice } from "../utils/spiritVoice.js";
import { buildFitnessBlock } from "../utils/fitnessEngine.js";

export const fitnessController = async (req, res) => {
  try {
    const { goal, experience, days, gender, height, weight } = req.body;

    if (!goal || !experience || !days) {
      return res.status(400).json({
        ok: false,
        error: "Missing required fields: goal, experience, days"
      });
    }

    // Build adaptive weekly block
    const block = buildFitnessBlock({
      goal,
      experience,
      days,
      gender,
      height,
      weight
    });

    // Apply unified Spirit voice
    const narration = spiritVoice("fitness", `Goal: ${goal}, Days: ${days}`);

    return res.json({
      ok: true,
      mode: "fitness",
      block,
      narration
    });
  } catch (err) {
    res.status(500).json({
      ok: false,
      error: err.message
    });
  }
};
