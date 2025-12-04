// routes/auth.js — Spirit v7.2 "Remember Me" System
import { Router } from "express";
import { loadUserMemory, saveUserMemory } from "../services/supabase.js";
import { randomUUID } from "crypto";

const router = Router();

/**
 * Request a magic login code
 */
router.post("/magic/start", async (req, res) => {
  const { email } = req.body;
  if (!email) return res.json({ ok: false, error: "Missing email." });

  const code = Math.floor(100000 + Math.random() * 900000).toString();

  await saveUserMemory(email, {
    magicCode: code,
    magicCodeIssuedAt: Date.now(),
  });

  return res.json({
    ok: true,
    message: "Magic code generated.",
    code, // in production you'd email it
  });
});

/**
 * Verify magic code and issue userId
 */
router.post("/magic/verify", async (req, res) => {
  const { email, code } = req.body;

  const { memory } = await loadUserMemory(email);

  if (!memory || memory.magicCode !== code) {
    return res.json({ ok: false, error: "Invalid code." });
  }

  const existingId =
    memory.userId || `spirit-${email}-${randomUUID().slice(0, 6)}`;

  await saveUserMemory(email, {
    userId: existingId,
    magicCode: null,
    magicCodeIssuedAt: null,
    lastLogin: Date.now(),
  });

  return res.json({ ok: true, userId: existingId });
});

/**
 * Check if Spirit remembers this email
 */
router.post("/magic/check", async (req, res) => {
  const { email } = req.body;

  const { memory } = await loadUserMemory(email);

  if (!memory?.userId) {
    return res.json({ ok: false, remembers: false });
  }

  return res.json({
    ok: true,
    remembers: true,
    userId: memory.userId,
  });
});

export default router;