// routes/health.js — Spirit v4.x Health Endpoints
// -----------------------------------------------
import express from "express";
import OpenAI from "openai";
import supabase from "../supabase.js";

const router = express.Router();

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// ─────────────────────────────────────────────
//  GET /health — lightweight status
// ─────────────────────────────────────────────
router.get("/", async (_req, res) => {
  const ts = new Date().toISOString();

  return res.status(200).json({
    ok: true,
    service: "Spirit v4.x",
    status: "online",
    ts,
  });
});

// ─────────────────────────────────────────────
//  GET /health/deep — async extended checks
//  (DB + OpenAI capability ping)
// ─────────────────────────────────────────────
router.get("/deep", async (_req, res) => {
  const ts = new Date().toISOString();

  // Respond immediately so platforms like Render don't time out
  res.status(202).json({
    ok: true,
    service: "Spirit v4.x",
    mode: "deep",
    note: "Running extended checks asynchronously",
    ts,
  });

  // Run extended checks in the background
  (async () => {
    try {
      // 1) Supabase check — can we read from reflections?
      let dbStatus = "unknown";
      try {
        const { error } = await supabase
          .from("reflections")
          .select("id")
          .limit(1);

        dbStatus = error ? "error" : "ok";
      } catch (err) {
        dbStatus = "error";
        console.error("[Deep Health] Supabase error:", err.message || err);
      }

      // 2) OpenAI check — can we list models?
      let aiStatus = "unknown";
      try {
        await client.models.list();
        aiStatus = "ok";
      } catch (err) {
        aiStatus = "error";
        console.error("[Deep Health] OpenAI error:", err.message || err);
      }

      console.log(
        `[Deep Health] ${ts} | DB: ${dbStatus} | OpenAI: ${aiStatus}`
      );
    } catch (err) {
      console.error("[Deep Health] Unexpected error:", err.message || err);
    }
  })();
});

export default router;