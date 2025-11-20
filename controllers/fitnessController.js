// controllers/fitnessController.js — Real Fitness Mode Logic v5.1

import supabase from "../supabase.js";
import { fitnessService } from "../services/fitnessService.js";

/* ============================================================
   POST /fitness/profile
   Save or update a user's fitness profile
   ============================================================ */
export const setFitnessProfile = async (req, res) => {
  try {
    const { user_id, fitness_profile } = req.body;

    if (!user_id || !fitness_profile) {
      return res.status(400).json({ ok: false, error: "Missing user_id or fitness_profile" });
    }

    // Upsert fitness profile
    const { data, error } = await supabase
      .from("fitness_profile")
      .upsert({
        user_id,
        goal: fitness_profile.goal,
        experience: fitness_profile.experience,
        commitment_days: fitness_profile.commitment_days,
        equipment: fitness_profile.equipment,
        height_cm: fitness_profile.height_cm,
        weight_kg: fitness_profile.weight_kg,
        limitations: fitness_profile.limitations || [],
        identity_mode: fitness_profile.identity,
        tone_mode: fitness_profile.tone,
        updated_at: new Date()
      })
      .select()
      .single();

    if (error) throw error;

    // Build a block if one doesn't exist yet
    const block = await fitnessService.ensureBlockExists(user_id, data);

    res.json({
      ok: true,
      message: "Fitness profile saved",
      profile: data,
      block
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
};


/* ============================================================
   GET /fitness/block
   Fetch user's current 4-week block
   ============================================================ */
export const getFitnessBlock = async (req, res) => {
  try {
    const user_id = req.query.user_id;

    const { data: block, error } = await supabase
      .from("fitness_blocks")
      .select("*")
      .eq("user_id", user_id)
      .order("created_at", { ascending: false })
      .limit(1);

    if (error) throw error;

    res.json({ ok: true, block: block[0] || null });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
};


/* ============================================================
   POST /fitness/block/regenerate
   Force creation of a new 4-week block
   ============================================================ */
export const generateFitnessBlock = async (req, res) => {
  try {
    const { user_id } = req.body;

    const { data: profile } = await supabase
      .from("fitness_profile")
      .select("*")
      .eq("user_id", user_id)
      .single();

    const block = await fitnessService.createBlock(user_id, profile);

    res.json({ ok: true, block });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
};


/* ============================================================
   GET /fitness/session/:day
   Generate today's session from block
   ============================================================ */
export const getSessionForDay = async (req, res) => {
  try {
    const user_id = req.query.user_id;
    const day = parseInt(req.params.day);

    const session = await fitnessService.generateSession(user_id, day);

    res.json({
      ok: true,
      day,
      session
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
};


/* ============================================================
   POST /fitness/session/log
   Save session history entry
   ============================================================ */
export const logSession = async (req, res) => {
  try {
    const { user_id, session } = req.body;

    const { error } = await supabase
      .from("session_logs")
      .insert({
        user_id,
        session_date: new Date(),
        session_title: session.title,
        session_json: session
      });

    if (error) throw error;

    res.json({ ok: true, message: "Session logged" });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
};