// index.js
// Spirit v7 — Production Entrypoint

import express from "express";
import cors from "cors";

import spiritRouter from "./routes/spirit.js";
import liveRouter from "./routes/live.js";
import debugRoutes from "./routes/debug.js";

import { hasSupabase } from "./services/supabase.js";

const PORT = process.env.PORT || 5000;

// ----------------------------------------------
// INITIALIZE APP
// ----------------------------------------------
const app = express();

app.use(cors());
app.use(express.json({ limit: "2mb" }));

// ----------------------------------------------
// HEALTH CHECK
// ----------------------------------------------
app.get("/", (req, res) => {
  res.json({
    ok: true,
    service: "Spirit v7",
    supabase: hasSupabase ? "connected" : "local-only",
  });
});

// ----------------------------------------------
// ROUTES
// ----------------------------------------------
app.use("/spirit", spiritRouter);
app.use("/live", liveRouter);
app.use("/debug", debugRoutes);

// ----------------------------------------------
// START SERVER
// ----------------------------------------------
app.listen(PORT, () => {
  console.log("✨ Spirit v7 Cognitive Engine running on port", PORT);
});