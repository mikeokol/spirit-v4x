// routes/health.js — Spirit v4.x Health Endpoints
// ------------------------------------------------
import express from "express";
import OpenAI from "openai";
import supabase from "../supabase.js";

const router = express.Router();
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

router.get("/", (_req, res) =>
  res.json({ ok: true, service: "Spirit v4.x", mode: "fast" })
);

router.get("/deep", async (_req, res) => {
  const ts = new Date().toISOString();

  // Return immediately so Render doesn't time out
  res.status(202).json({
    ok: true,
    service: "Spirit v4.x",
    mode: "deep",
    ts,
  });

  // Background checks
  (async () => {
    try {
      const { error } = await supabase.from("reflections").select("id").limit(1);
      const dbStatus = error ? "error" : "ok";

      const aiStatus = await client.models
        .list()
        .then(() => "ok")
        .catch(() => "error");

      console.log(`[Deep Health] ${ts} DB:${dbStatus} | OpenAI:${aiStatus}`);
    } catch (err) {
      console.error("[Deep Health] Error:", err.message);
    }
  })();
});

export default router;