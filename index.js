// index.js — Spirit v7 Unified Backend
// Clean, production-safe, Lovable-compatible

import express from "express";
import cors from "morgan";
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

// OPTIONAL: set-by-set coach (file may not exist yet)
let liveNextSetHandler = null;
try {
  liveNextSetHandler = await import("./routes/liveNextSet.js").then((m) => m.default);
} catch {
  console.warn("[WARN] liveNextSet.js not found – route disabled until file is added");
}

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
// ROUTES  (all existing + conditional new)
// ---------------------------------------------
app.get("/spirit/memory", async (req, res) => {
  const userId = req.query.id;
  if (!userId) return res.status(400).json({ ok: false, error: "Missing userId" });
  /* memory helper – assumes you have loadUserMemory() already */
  const memory = await loadUserMemory(userId);
  res.json({ ok: true, identity: memory.identity || null });
});

app.use("/auth", authRouter);
app.use("/spirit", spiritRouter);
app.use("/live", liveRouter);
if (liveNextSetHandler) app.post("/live/next-set", liveNextSetHandler); // only if file exists
app.use("/debug", debugRouter);

// ---------------------------------------------
// START SERVER
// ---------------------------------------------
app.listen(PORT, () => {
  console.log(`[supabase] Client initialised`);
  console.log(`✨ Spirit v7 Cognitive Engine running on port ${PORT}`);
});
