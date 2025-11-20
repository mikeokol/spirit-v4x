// chat.js — Spirit v5.1 Chat Router
// ---------------------------------

import express from "express";
import { handleChat } from "./routes/ChatController.js";

const router = express.Router();

router.post("/", async (req, res) => {
  console.log("🟣 [CHAT] Incoming request:", {
    prompt: req.body?.prompt,
    userId: req.body?.userId,
  });

  return handleChat(req, res);
});

export default router;