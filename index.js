// index.js — Spirit v5.1 Server Entrypoint
// ------------------------------------------------------

import "dotenv/config";
import express from "express";
import cors from "cors";

import healthRouter from "./routes/health.js";
import chatRouter from "./chat.js"; // wrapper for ChatController.js

// ─────────────────────────────────────────────
//  Initialize App
// ─────────────────────────────────────────────
const app = express();

// ─────────────────────────────────────────────
//  CORS — allow Lovable + localhost
// ─────────────────────────────────────────────
app.use(
  cors({
    origin: [
      "http://localhost:3000",
      "https://spirit-ai-coach-creator.lovable.app",
      /\.lovable\.app$/,
    ],
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

// ─────────────────────────────────────────────
//  Middleware
// ─────────────────────────────────────────────
app.use(express.json());

// ─────────────────────────────────────────────
//  Routes
// ─────────────────────────────────────────────
app.use("/health", healthRouter);
app.use("/chat", chatRouter);

// Root route — simple sanity check
app.get("/", (req, res) => {
  res.status(200).json({
    ok: true,
    service: "Spirit v5.1",
    message: "You have arrived. Breathe. We begin.",
    ts: new Date().toISOString(),
  });
});

// ─────────────────────────────────────────────
//  Start Server
// ─────────────────────────────────────────────
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`🔥 Spirit backend running on port ${PORT}`);
});