// services/reflectionService.js — Reflection Logic v5.1

import supabase from "../supabase.js";
import { spiritClient } from "./spiritClient.js";

import { systemPrompt } from "../prompts/systemPrompt.js";
import { reflectionPrompt } from "../prompts/reflectionPrompt.js";

export const reflectionService = {
  generateReflection: async ({ user_id, intention, emotion, identity_focus }) => {
    const payload = {
      intention: intention || "",
      emotion: emotion || "",
      identity_focus: identity_focus || "identity"
    };

    const raw = await spiritClient.callModel(
      systemPrompt,
      reflectionPrompt,
      payload
    );

    if (!raw || !raw.reflection) {
      throw new Error("Invalid reflection format from model");
    }

    const r = raw.reflection;

    const { data, error } = await supabase
      .from("reflections")
      .insert({
        user_id,
        intention,
        emotion,
        mirror: r.mirror,
        core_insight: r.core_insight,
        corrective_path: r.corrective_path,
        identity_anchor: r.identity_anchor
      })
      .select()
      .single();

    if (error) throw error;

    return data;
  }
};