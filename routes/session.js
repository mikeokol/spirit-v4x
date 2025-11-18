// routes/session.js — Spirit v5.0 Session Management API
// -------------------------------------------------------
import express from "express";
import supabase from "../supabase.js";

const router = express.Router();

/**
 * Normalize userId from any source: header, query, body.
 */
function resolveUserId(req) {
  return (
    req.headers["x-user-id"] ||
    req.body?.userId ||
    req.query?.userId ||
    null
  );
}

/**
 * Ensure session row exists for a new user.
 */
async function ensureSession(userId) {
  if (!userId) return null;

  const now = new Date().toISOString();

  // Check if session already exists
  const { data: existing } = await supabase
    .from("sessions")
    .select("*")
    .eq("user_id", userId)
    .maybeSingle();

  if (existing) return existing;

  // Create new empty session
  const { data: created } = await supabase
    .from("sessions")
    .insert({
      user_id: userId,
      training_block: null,
      training_day: 1,
      last_mode: "sanctuary",
      last_intention: null,
      created_at: now,
      updated_at: now
    })
    .select()
    .maybeSingle();

  return created;
}

// ---------------------------------------------------------------------
// GET /session — return full session data for this user
// ---------------------------------------------------------------------
router.get("/", async (req, res) => {
  const userId = resolveUserId(req);

  if (!userId) {
    return res.status(400).json({
      ok: false,
      error: "Missing userId for session lookup",
    });
  }

  try {
    const session = await ensureSession(userId);

    return res.json({
      ok: true,
      session,
      ts: new Date().toISOString()
    });
  } catch (err) {
    console.error("[session/get] error:", err.message);
    return res.status(500).json({
      ok: false,
      error: "Session retrieval failed.",
      details: err.message
    });
  }
});

// ---------------------------------------------------------------------
// POST /session/update — update last_mode or last_intention
// Body: { userId, lastMode, lastIntention }
// ---------------------------------------------------------------------
router.post("/update", async (req, res) => {
  const { userId, lastMode, lastIntention } = req.body || {};

  if (!userId) {
    return res.status(400).json({
      ok: false,
      error: "Missing userId"
    });
  }

  const now = new Date().toISOString();

  try {
    const { data, error } = await supabase
      .from("sessions")
      .update({
        last_mode: lastMode || null,
        last_intention: lastIntention || null,
        updated_at: now
      })
      .eq("user_id", userId)
      .select()
      .maybeSingle();

    if (error) throw error;

    return res.json({
      ok: true,
      updated: data,
      ts: now
    });
  } catch (err) {
    console.error("[session/update] error:", err.message);
    return res.status(500).json({
      ok: false,
      error: "Session update failed",
      details: err.message
    });
  }
});

// ---------------------------------------------------------------------
// POST /session/block — store or update training_block
// Body: { userId, trainingBlock }
// ---------------------------------------------------------------------
router.post("/block", async (req, res) => {
  const { userId, trainingBlock } = req.body || {};

  if (!userId || !trainingBlock) {
    return res.status(400).json({
      ok: false,
      error: "Missing userId or trainingBlock"
    });
  }

  const now = new Date().toISOString();

  try {
    const { data, error } = await supabase
      .from("sessions")
      .update({
        training_block: trainingBlock,
        updated_at: now
      })
      .eq("user_id", userId)
      .select()
      .maybeSingle();

    if (error) throw error;

    return res.json({
      ok: true,
      block: data,
      ts: now
    });
  } catch (err) {
    console.error("[session/block] error:", err.message);
    return res.status(500).json({
      ok: false,
      error: "Failed to update training block.",
      details: err.message
    });
  }
});

export default router;
