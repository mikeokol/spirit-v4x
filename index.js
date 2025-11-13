// index.js — Spirit Core v3.3 (Server Entrypoint)
// ------------------------------------------------------

import express from "express";
import bodyParser from "body-parser";
import "dotenv/config";

import supabase from "./supabase.js";
import healthRouter from "./routes/health.js";
import chatRouter from "./routes/chat.js";

// ─────────────────────────────────────────────
//  Setup
// ─────────────────────────────────────────────
const app = express();
app.use(bodyParser.json());

// ─────────────────────────────────────────────
//  Routes
// ─────────────────────────────────────────────
app.use("/health", healthRouter);
app.use("/chat", chatRouter);

// Root route — sanity check
app.get("/", (_req, res) => {
  res.status(200).json({
    ok: true,
    service: "Spirit v3.3",
    message: "You have arrived. Breathe. We begin.",
    ts: new Date().toISOString(),
  });
});

// ─────────────────────────────────────────────
//  Start Server
// ─────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`✅ Spirit API running on port ${PORT}`));

export default app;