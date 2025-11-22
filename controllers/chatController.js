// chatController.js — Spirit v5.1 Showcase Brain

import { classifyIntent } from "../utils/classifyIntent.js";
import { spiritVoice } from "../utils/spiritVoice.js";

export const chatController = async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) {
      return res.status(400).json({ ok: false, error: "Missing message" });
    }

    const intent = classifyIntent(message);

    switch (intent) {
      case "fitness":
        return res.json({
          ok: true,
          mode: "fitness",
          reply: spiritVoice("fitness", message)
        });

      case "creator":
        return res.json({
          ok: true,
          mode: "creator",
          reply: spiritVoice("creator", message)
        });

      case "hybrid":
        return res.json({
          ok: true,
          mode: "hybrid",
          reply: spiritVoice("hybrid", message)
        });

      case "live":
        return res.json({
          ok: true,
          mode: "live",
          reply: spiritVoice("live", message)
        });

      case "reflection":
        return res.json({
          ok: true,
          mode: "reflection",
          reply: spiritVoice("reflection", message)
        });

      default:
        return res.json({
          ok: true,
          mode: "general",
          reply: spiritVoice("general", message)
        });
    }
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
};
