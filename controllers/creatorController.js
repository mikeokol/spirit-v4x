// controllers/creatorController.js — Creator Engine v5.1

import supabase from "../supabase.js";
import { creatorService } from "../services/creatorService.js";

/* ============================================================
   POST /creator/profile
   Save or update a creator profile
   ============================================================ */
export const setCreatorProfile = async (req, res) => {
  try {
    const { user_id, creator_profile } = req.body;

    if (!user_id || !creator_profile) {
      return res.status(400).json({ ok: false, error: "Missing user_id or creator_profile" });
    }

    const { data, error } = await supabase
      .from("creator_profile")
      .upsert({
        user_id,
        niche: creator_profile.niche,
        identity: creator_profile.identity,
        platform: creator_profile.platform,
        voice: creator_profile.voice,
        style: creator_profile.style,
        updated_at: new Date()
      })
      .select()
      .single();

    if (error) throw error;

    res.json({
      ok: true,
      message: "Creator profile saved",
      profile: data
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
};


/* ============================================================
   POST /creator/script
   Generate a world-class script + structure
   ============================================================ */
export const generateCreatorScript = async (req, res) => {
  try {
    const { user_id, topic, overrides } = req.body;

    if (!user_id || !topic) {
      return res.status(400).json({ ok: false, error: "Missing user_id or topic" });
    }

    // Get creator profile
    const { data: profile, error: profileErr } = await supabase
      .from("creator_profile")
      .select("*")
      .eq("user_id", user_id)
      .single();

    if (profileErr) throw profileErr;

    const scriptPayload = await creatorService.generateScript(profile, { topic, overrides });

    res.json({
      ok: true,
      script: scriptPayload
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
};


/* ============================================================
   POST /creator/sprint
   Generate a 7-day or similar content sprint
   ============================================================ */
export const generateContentSprint = async (req, res) => {
  try {
    const { user_id, sprint_config } = req.body;

    if (!user_id) {
      return res.status(400).json({ ok: false, error: "Missing user_id" });
    }

    const { data: profile, error: profileErr } = await supabase
      .from("creator_profile")
      .select("*")
      .eq("user_id", user_id)
      .single();

    if (profileErr) throw profileErr;

    const sprint = await creatorService.generateSprint(profile, sprint_config || {});

    res.json({
      ok: true,
      sprint
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
};