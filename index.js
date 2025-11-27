// index.js — Spirit v7 Backend (Unified Engine + Live Sessions)

import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import "dotenv/config";

import spiritRouter from "./routes/spirit.js";
import liveRouter from "./routes/live.js";

const app = express();

// =============================================================
// GLOBAL MIDDLEWARE
// =============================================================
app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  })
);

app.use(
  bodyParser.json({
    limit: "5mb",
    strict: true,
  })
);

// =============================================================
// ROUTES
// =============================================================

// Unified Cognitive Engine
app.use("/spirit", spiritRouter);

// Live coaching sessions (start / message / end)
app.use("/live", liveRouter);

// Root health
app.get("/", (_req, res) => {
  res.json({
    ok: true,
    engine: "Spirit v7 Cognitive Engine",
    status: "online",
    message: "You have arrived. Breathe. Spirit v7 is awake.",
    endpoints: {
      unified: "/spirit",
      live: {
        start: "/live/start",
        message: "/live/message",
        end: "/live/end",
      },
    },
  });
});

// 404 fallback
app.use((req, res) => {
  res.status(404).json({
    ok: false,
    error: "Route not found.",
    path: req.originalUrl,
  });
});

// =============================================================
// SERVER STARTUP
// =============================================================
const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`✨ Spirit v7 Cognitive Engine running on port ${PORT}`);
});