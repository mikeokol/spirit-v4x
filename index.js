// index.js — Spirit v7 Unified Backend
// Clean, production-safe, Lovable-compatible

import express from "express";
import cors from "cors";
import morgan from "morgan";
import dotenv from "dotenv";

dotenv.config();

// ---------------------------------------------
// ROUTERS
// ---------------------------------------------
import authRouter from "./routes/auth.js";
import spiritRouter from "./routes/spirit.js";
import liveRouter from "./routes/live.js";
import debugRouter from "./routes/debug.js";

// ---------------------------------------------
// INIT APP
// ---------------------------------------------
const app = express();
const PORT = process.env.PORT || 5000;

// ---------------------------------------------
// MIDDLEWARE
// ---------------------------------------------
app.use(cors({
  origin: "*",           // Lovable frontend
  methods: "GET,POST",
  allowedHeaders: "Content-Type, Authorization",
}));

app.use(express.json({ limit: "2mb" }));
app.use(morgan("dev"));   // logs requests

// ---------------------------------------------
// HEALTH CHECK
// ---------------------------------------------
app.get("/", (req, res) => {
  res.json({
    ok: true,
    service: "Spirit v7",
    status: "running",
    supabase: "connected",
  });
});

// ---------------------------------------------
// ROUTES
// ---------------------------------------------
app.use("/auth", authRouter);     // email login / magic link
app.use("/spirit", spiritRouter); // creator, fitness, reflection, hybrid
app.use("/live", liveRouter);     // live coaching flow + AI hybrid
app.use("/debug", debugRouter);   // debugging endpoints

// ---------------------------------------------
// START SERVER
// ---------------------------------------------
app.listen(PORT, () => {
  console.log(`[supabase] Client initialised`);
  console.log(`✨ Spirit v7 Cognitive Engine running on port ${PORT}`);
});