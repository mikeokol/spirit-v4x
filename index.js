// index.js — Spirit v5.0 Server Entrypoint
// -----------------------------------------
import "dotenv/config";
import express from "express";
import cors from "cors";

import healthRouter from "./routes/health.js";
import chatRouter from "./routes/chat.js";
import liveRouter from "./routes/live.js";

// ─────────────────────────────────────────────
//  Initialize
// ─────────────────────────────────────────────
const app = express();

// ─────────────────────────────────────────────
//  CORS — allow Lovable, localhost, and preview links
// ─────────────────────────────────────────────
app.use(
  cors({
    origin: [
      "http://localhost:3000",
      "http://localhost:5173",
      /\.lovable\.app$/, // Allow ANY Lovable preview or deployed domain
    ],
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

// ─────────────────────────────────────────────
//  Middleware
// ─────────────────────────────────────────────
app.use(express.json({ limit: "2mb" }));

// ─────────────────────────────────────────────
//  Routes
// ─────────────────────────────────────────────
app.use("/health", healthRouter);
app.use("/chat", chatRouter);
app.use("/live", liveRouter);

// Root sanity check
app.get("/", (_req, res) => {
  res.status(200).json({
    ok: true,
    service: "Spirit v5.0",
    message: "You have arrived. Breathe. We begin.",
    ts: new Date().toISOString(),
  });
});

// ─────────────────────────────────────────────
//  Start Server
// ─────────────────────────────────────────────
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`🔥 Spirit v5.0 active on port ${PORT}`);
});
