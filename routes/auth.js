// routes/auth.js
// Spirit v7 — Email Magic Link Auth Handler

import express from "express";
import { supabase, hasSupabase } from "../services/supabase.js";

const router = express.Router();

// ---------------------------------------------------------
// CHECK ENV
// ---------------------------------------------------------
if (!hasSupabase) {
  console.warn("[AUTH] Supabase keys missing — auth disabled.");
}

// ---------------------------------------------------------
// SEND MAGIC LINK
// ---------------------------------------------------------
router.post("/login", async (req, res) => {
  try {
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({ ok: false, error: "Missing email" });
    }

    if (!hasSupabase) {
      return res.status(500).json({ ok: false, error: "Supabase not configured" });
    }

    const { data, error } = await supabase.auth.signInWithOtp({
      email,
      options: {
        emailRedirectTo: process.env.SITE_URL || "http://localhost:3000"
      }
    });

    if (error) {
      console.error("[AUTH error]", error);
      return res.status(500).json({ ok: false, error: error.message });
    }

    return res.json({ ok: true, message: "Magic link sent" });
  } catch (err) {
    console.error("[AUTH /login error]", err);
    return res.status(500).json({ ok: false, error: "Login failed" });
  }
});

// ---------------------------------------------------------
// VALIDATE SESSION (Frontend sends token to backend)
// ---------------------------------------------------------
router.post("/verify", async (req, res) => {
  try {
    const { access_token } = req.body;

    if (!access_token) {
      return res.status(400).json({ ok: false, error: "Missing access_token" });
    }

    const { data, error } = await supabase.auth.getUser(access_token);

    if (error || !data?.user) {
      return res.status(401).json({ ok: false, error: "Invalid token" });
    }

    // This userId ties the tester to ALL Spirit memory systems
    const userId = data.user.id;

    return res.json({
      ok: true,
      userId,
      email: data.user.email
    });

  } catch (err) {
    console.error("[AUTH /verify error]", err);
    return res.status(500).json({ ok: false, error: "Verification failed" });
  }
});

export default router;