// services/fitnessService.js — Fitness Engine v5.1

import { systemPrompt } from "../prompts/systemPrompt.js";
import { fitnessPrompt } from "../prompts/fitnessPrompt.js";

export const fitnessService = {
  
  /* ----------------------------------------------------------
     Build new 4-week block
     ---------------------------------------------------------- */
  createBlock: async (user_id, profile) => {
    const aiPayload = { fitness_profile: profile };

    const raw = await spiritClient.callModel(fitnessPrompt, aiPayload);

    if (!raw || !raw.block) throw new Error("AI returned invalid block");

    const { data, error } = await supabase
      .from("fitness_blocks")
      .insert({
        user_id,
        phase: raw.block.phase,
        focus: raw.block.focus,
        identity_anchor: raw.block.identity_anchor,
        weekly_structure: raw.block.weekly_structure,
        progression_plan: raw.block.progression_plan
      })
      .select()
      .single();

    if (error) throw error;

    return data;
  },


  /* ----------------------------------------------------------
     If no block exists, create one automatically
     ---------------------------------------------------------- */
  ensureBlockExists: async (user_id, profile) => {
    const { data: blocks } = await supabase
      .from("fitness_blocks")
      .select("*")
      .eq("user_id", user_id)
      .order("created_at", { ascending: false })
      .limit(1);

    if (blocks.length > 0) return blocks[0];

    return await fitnessService.createBlock(user_id, profile);
  },


  /* ----------------------------------------------------------
     Generate a Training Session For a Day (1–7)
     ---------------------------------------------------------- */
  generateSession: async (user_id, day) => {
    const { data: block } = await supabase
      .from("fitness_blocks")
      .select("*")
      .eq("user_id", user_id)
      .order("created_at", { ascending: false })
      .limit(1);

    if (!block || !block.length) throw new Error("No training block found");

    const aiPayload = {
      block: block[0],
      day
    };

    const raw = await spiritClient.callModel(
  systemPrompt,
  fitnessPrompt,
  aiPayload
);

    if (!raw || !raw.session) throw new Error("AI returned invalid session");

    return raw.session;
  }
};