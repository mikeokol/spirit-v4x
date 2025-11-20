// controllers/reflectionController.js — Reflection Engine v5.1

import { reflectionService } from "../services/reflectionService.js";
import supabase from "../supabase.js";

export const generateReflection = async (req, res) => {
  try {
    const { user_id, intention, emotion, identity_focus } = req.body;

    if (!user_id) {
      return res.status(400).json({ ok: false, error: "Missing user_id" });
    }

    const reflection = await reflectionService.generateReflection({
      user_id,
      intention,
      emotion,
      identity_focus
    });

    res.json({
      ok: true,
      reflection
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
};

export const getReflectionHistory = async (req, res) => {
  try {
    const user_id = req.query.user_id;

    if (!user_id) {
      return res.status(400).json({ ok: false, error: "Missing user_id" });
    }

    const { data, error } = await supabase
      .from("reflections")
      .select("*")
      .eq("user_id", user_id)
      .order("created_at", { ascending: false })
      .limit(30);

    if (error) throw error;

    res.json({
      ok: true,
      history: data
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
};