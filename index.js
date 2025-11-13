// index.js — Spirit Core v4.x (Server Entrypoint)
// ------------------------------------------------------

import express from "express";
import cors from "cors";
import "dotenv/config";

import healthRouter from "./routes/health.js";
import chatRouter from "./routes/chat.js";

// ─────────────────────────────────────────────
//  Setup
// ─────────────────────────────────────────────
const app = express();

// JSON body parser
app.use(express.json({ limit: "1mb" }));

// CORS — required for Lovable frontend
app.use(
  cors({
    origin: "*", // allow all frontend origins
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

// ─────────────────────────────────────────────
//  Routes
// ─────────────────────────────────────────────
app.use("/health", healthRouter);
app.use("/chat", chatRouter);

// Root route — sanity check
app.get("/", (_req, res) => {
  res.status(200).json({
    ok: true,
    service: "Spirit v4.x",
    message: "You have arrived. Spirit is online.",
    ts: new Date().toISOString(),
  });
});

// ─────────────────────────────────────────────
//  Start Server
// ─────────────────────────────────────────────
const PORT = process.env.PORT || 10000;
app.listen(PORT, () =>
  console.log(`✅ Spirit API (v4.x) running on port ${PORT}`)
);

export default app;
