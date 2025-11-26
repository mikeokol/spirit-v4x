// index.js — Spirit v7 Cognitive Engine Backend (Render-ready)

import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import "dotenv/config";

// Legacy mode routes (still active)
import chatRouter from "./routes/chat.js";
import fitnessRouter from "./routes/fitness.js";
import creatorRouter from "./routes/creator.js";
import hybridRouter from "./routes/hybrid.js";
import liveRouter from "./routes/live.js";
import healthRouter from "./routes/health.js";
import analyticsRouter from "./routes/analytics.js";
import voiceRouter from "./routes/voice.js";

// NEW — Unified Cognitive Engine
// Must be the v7 version with controller → planner → executor → critic
import spiritRouter from "./routes/spirit.js";

const app = express();

// =============================================================
// GLOBAL MIDDLEWARE (Render-friendly)
// =============================================================
app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  })
);

app.use(
  bodyParser.json({
    limit: "10mb",
    strict: false, // required for planner tree + nested objects
  })
);

// =============================================================
// ROUTES
// =============================================================
app.use("/health", healthRouter);

// Legacy mode endpoints
app.use("/chat", chatRouter);
app.use("/fitness", fitnessRouter);
app.use("/creator", creatorRouter);
app.use("/hybrid", hybridRouter);
app.use("/live-session", liveRouter);
app.use("/analytics", analyticsRouter);
app.use("/voice", voiceRouter);

// Unified Cognitive Engine v7
app.use("/spirit", spiritRouter);

// Root
app.get("/", (_req, res) => {
  res.json({
    ok: true,
    engine: "Spirit v7 Cognitive Engine",
    version: "7.0",
    status: "online",
    message: "You have arrived. The Trinity v7 engine is active.",
    routes: {
      unified: "/spirit",
      legacy: [
        "/chat",
        "/fitness",
        "/creator",
        "/hybrid",
        "/live-session",
        "/analytics",
        "/voice",
      ],
    },
  });
});

// 404
app.use((req, res) => {
  res.status(404).json({
    ok: false,
    error: "Route not found",
    path: req.originalUrl,
  });
});

// =============================================================
// SERVER — Render requires PORT binding correctly
// =============================================================
const PORT = process.env.PORT || 10000;

app.listen(PORT, () => {
  console.log(`✨ Spirit v7 Cognitive Engine running on port ${PORT}`);
});