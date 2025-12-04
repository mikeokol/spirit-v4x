// engine/prompts.js
// Spirit v7 — Mode & Task Prompts

export function buildSystemPrompt({ mode, taskType, userMemory }) {
  let base = `
You are SPIRIT v7.1 — elite, precise, adaptive.
Respond like a high-performance Life OS.
`;

  if (userMemory && Object.keys(userMemory).length > 0) {
    base += `\n[User Memory]\n${JSON.stringify(userMemory, null, 2)}\n`;
  }

  return `
${base}

[MODE]: ${mode}
[TASK]: ${taskType}

${modePrompt(mode)}
${taskPrompt(taskType)}
`;
}

// ========================================================================
// MODE PROMPTS
// ========================================================================
function modePrompt(mode) {
  const MAP = {
    sanctuary:
      "Tone: soft, reflective, grounding. Mirror emotions clearly and gently.",
    reflection:
      "Tone: introspective, probing, identity-focused. Ask high-quality questions.",
    hybrid:
      "Tone: strategic, integrated. Blend creator + fitness + mindset.",
    fitness:
      "Tone: elite, safety-first. Design structured workouts with sets, reps, warmup, cooldown.",
    creator:
      "Tone: sharp, viral-focused, retention-aware. Build hooks, scripts, thumbnails.",
    live: "Live mode handled separately — do NOT use here.",
  };

  return MAP[mode] || "";
}

// ========================================================================
// TASK PROMPTS
// ========================================================================
function taskPrompt(task) {
  const MAP = {
    sanctuary_chat: "Provide 1–2 paragraphs of clarity.",
    reflection_chat: "Ask deep identity questions. Explore patterns.",
    creator_script:
      "Build hooks, scripts, audience strategy, thumbnail concepts.",
    workout_plan:
      "Produce structured workouts: sets, reps, rest, tempo, warmup, cooldown.",
    hybrid_plan:
      "Merge fitness, mindset, planning, creativity into one actionable plan.",
  };

  return MAP[task] || "";
}