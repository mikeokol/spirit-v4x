// controllers/chatController.js — Unified Sanctuary Router (Spirit v5.1)

import { classifyIntent } from "../services/classifier.js";
import { generateReflection } from "./reflectionController.js";

export const chatController = async (req, res) => {
  try {
    const { user_id, message } = req.body;

    if (!user_id || !message) {
      return res.status(400).json({
        ok: false,
        error: "Missing user_id or message"
      });
    }

    const intent = classifyIntent(message);

    switch (intent) {
      case "fitness":
        return res.json({
          ok: true,
          mode: "fitness",
          next: "/fitness/profile or /fitness/session",
          note: "Detected fitness intent"
        });

      case "creator":
        return res.json({
          ok: true,
          mode: "creator",
          next: "/creator/script or /creator/profile",
          note: "Detected creator intent"
        });

      case "reflection":
        req.body.intention = message;
        return generateReflection(req, res);

      default:
        return res.json({
          ok: true,
          mode: "general",
          reply: "State your intention. Spirit is present."
        });
    }
  } catch (err) {
    res.status(500).json({
      ok: false,
      error: err.message
    });
  }
};