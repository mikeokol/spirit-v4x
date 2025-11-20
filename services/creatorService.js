// services/creatorService.js — Creator Logic v5.1

import { systemPrompt } from "../prompts/systemPrompt.js";
import { creatorPrompt } from "../prompts/creatorPrompt.js";

export const creatorService = {
  /* ----------------------------------------------------------
     Generate a world-class script for short/long-form
     ---------------------------------------------------------- */
  generateScript: async (profile, params) => {
    const { topic, overrides = {} } = params;

    const payload = {
      mode: "script",
      creator_profile: profile,
      request: {
        topic,
        platform: overrides.platform || profile.platform,
        voice: overrides.voice || profile.voice,
        style: overrides.style || profile.style,
        length: overrides.length || "shortform",
        goal: overrides.goal || "grow audience"
      }
    };

    const raw = await spiritClient.callModel(
  systemPrompt,
  creatorPrompt,
  payload
);

    if (!raw || !raw.creator_script) {
      throw new Error("AI returned invalid creator_script payload");
    }

    const script = raw.creator_script;

    const { data, error } = await supabase
      .from("creator_content")
      .insert({
        user_id: profile.user_id,
        hook: script.hook,
        script: script.script,
        beats: script.beats,
        visual_cues: script.visual_cues,
        posting_strategy: script.posting_strategy
      })
      .select()
      .single();

    if (error) throw error;

    return data;
  },

  /* ----------------------------------------------------------
     Generate a 7-day content sprint
     ---------------------------------------------------------- */
  generateSprint: async (profile, sprintConfig) => {
    const payload = {
      mode: "sprint",
      creator_profile: profile,
      request: {
        length_days: sprintConfig.length_days || 7,
        theme: sprintConfig.theme || null,
        goal: sprintConfig.goal || "grow audience"
      }
    };

    const raw = await spiritClient.callModel(creatorPrompt, payload);

    if (!raw || !raw.content_sprint) {
      throw new Error("AI returned invalid content_sprint payload");
    }

    const sprint = raw.content_sprint;

    const { data, error } = await supabase
      .from("content_sprints")
      .insert({
        user_id: profile.user_id,
        sprint_json: sprint
      })
      .select()
      .single();

    if (error) throw error;

    return data;
  }
};