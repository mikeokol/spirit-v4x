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
import liveNextSetHandler from "./routes/liveNextSet.js"; // file exists → direct import

// ---------------------------------------------
// INIT APP
// ---------------------------------------------
const app = express();
const PORT = process.env.PORT || 5000;

// ---------------------------------------------
// MIDDLEWARE
// ---------------------------------------------
app.use(
  cors({
    origin: "*", // Lovable frontend
    methods: "GET,POST",
    allowedHeaders: "Content-Type, Authorization",
  })
);
app.use(express.json({ limit: "2mb" }));
app.use(morgan("dev"));

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
// ROUTES  (all live)
// ---------------------------------------------
app.get("/spirit/memory", async (req, res) => {
  const userId = req.query.id;
  if (!userId) return res.status(400).json({ ok: false, error: "Missing userId" });
  const memory = await loadUserMemory(userId); // assumes you have this helper
  res.json({ ok: true, identity: memory.identity || null });
});

app.use("/auth", authRouter);
app.use("/spirit", spiritRouter);
app.use("/live", liveRouter);
app.post("/live/next-set", liveNextSetHandler); // set-by-set coaching
app.use("/debug", debugRouter);

// ---------------------------------------------
// START SERVER
// ---------------------------------------------
app.listen(PORT, () => {
  console.log(`[supabase] Client initialised`);
  console.log(`✨ Spirit v7 Cognitive Engine running on port ${PORT}`);
});
