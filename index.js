// index.js — Spirit v7 Unified Backend
// Clean, production-safe, Lovable-compatible + polish

import express from "express";
import cors from "cors";
import morgan from "morgan";
import dotenv from "dotenv";
dotenv.config();

import authRouter from "./routes/auth.js";
import spiritRouter from "./routes/spirit.js";
import liveRouter from "./routes/live.js";
import debugRouter from "./routes/debug.js";
import liveNextSetHandler from "./routes/liveNextSet.js";
import resumeHandler from "./routes/resume.js"; // ← PRP
import { clearECM } from "./engine/clearECM.js"; // ← Start fresh

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors({ origin: "*", methods: "GET,POST", allowedHeaders: "Content-Type, Authorization" }));
app.use(express.json({ limit: "2mb" }));
app.use(morgan("dev"));

app.get("/", (req, res) => res.json({ ok: true, service: "Spirit v7", status: "running", supabase: "connected" }));

// memory ribbon helper
app.get("/spirit/memory", async (req, res) => {
  const userId = req.query.id;
  if (!userId) return res.status(400).json({ ok: false, error: "Missing userId" });
  const memory = await loadMemory(userId); // your existing helper
  res.json({ ok: true, identity: memory.identity || null });
});

app.use("/auth", authRouter);
app.use("/spirit", spiritRouter);
app.use("/live", liveRouter);
app.post("/live/next-set", liveNextSetHandler); // set-by-set coaching
app.get("/spirit/resume", resumeHandler); // ← PRP endpoint
app.post("/spirit/clear-ecm", async (req, res) => { // ← Start fresh button
  const { userId } = req.body;
  if (!userId) return res.status(400).json({ ok: false });
  await clearECM(userId);
  res.json({ ok: true });
});
app.use("/debug", debugRouter);

app.listen(PORT, () => console.log(`✨ Spirit v7 Cognitive Engine running on port ${PORT}`));
