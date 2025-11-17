// routes/health.js — Spirit v5.0 Prep: Clean Health + Diagnostics Layer
// --------------------------------------------------------------------
import express from "express";
import OpenAI from "openai";
import supabase from "../supabase.js";

const router = express.Router();

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// --------------------------------------------------------------------
// GET /health — lightweight service heartbeat
// --------------------------------------------------------------------
router.get("/", async (_req, res) => {
  const ts = new Date().toISOString();

  return res.status(200).json({
    ok: true,
    service: "Spirit",
    status: "online",
    version: "v5.0-prep",
    ts,
  });
});

// --------------------------------------------------------------------
// GET /health/deep — async extended checks
// (DB connectivity + OpenAI model availability)
// --------------------------------------------------------------------
router.get("/deep", async (_req, res) => {
  const ts = new Date().toISOString();

  // Respond early to avoid Render / deployment timeouts
  res.status(202).json({
    ok: true,
    service: "Spirit",
    mode: "deep-check",
    note: "Running extended checks asynchronously",
    version: "v5.0-prep",
    ts,
  });

  // Background deep diagnostic
  (async () => {
    try {
      // ─────────────────────────────────────────────
      // 1) Supabase check — test reflections table
      // ─────────────────────────────────────────────
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

      // ─────────────────────────────────────────────
      // 2) OpenAI check — ensure model listing works
      // ─────────────────────────────────────────────
      let aiStatus = "unknown";
      try {
        await client.models.list();
        aiStatus = "ok";
      } catch (err) {
        aiStatus = "error";
        console.error("[Deep Health] OpenAI error:", err.message || err);
      }

      console.log(
        `[Deep Health] ${ts} | DB: ${dbStatus} | AI: ${aiStatus}`
      );
    } catch (err) {
      console.error("[Deep Health] Unexpected diagnostic error:", err.message || err);
    }
  })();
});

export default router;
