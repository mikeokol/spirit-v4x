// engine/liveCoachVoice.js — Spirit v7.2 Voice Layer for Live Coaching

import { speak, speakRest, getMotivationCue } from "../services/voice.js";

export async function voiceWarmupBlock() {
  return speak(
    "Let's begin your warmup. Three minutes of light movement — walking, cycling, or an easy jog. Focus on breathing and loosening the joints."
  );
}

export async function voiceMainBlock(exercise, duration, intensity = 5) {
  const cue = getMotivationCue(intensity);

  return speak(
    `${exercise} for ${duration} seconds. ${cue}`
  );
}

export async function voiceCountdown(seconds = 5) {
  return speak(`Starting in ${seconds} seconds. Get into position.`);
}

/**
 * Safety layer — Spirit always includes this if user intensity spikes
 */
export function safetyLine() {
  return "Remember: if you feel sharp pain, dizziness, or joint discomfort, stop immediately and reset. Safety first.";
}

/**
 * Gender-aware scaling for recommended weights
 */
export function scaledWeight(base, gender) {
  if (gender === "female") return Math.max(2.5, Math.floor(base * 0.65));
  if (gender === "male") return base;
  return Math.floor(base * 0.8); // fallback neutral
}

/**
 * Convert a fitness-plan block into a voice-friendly coaching script
 */
export function convertPlanToVoiceBlocks(planText) {
  const lines = planText.split("\n").map((l) => l.trim());

  const blocks = [];

  for (const line of lines) {
    if (line.startsWith("- Warmup:")) {
      blocks.push({
        type: "warmup",
        text: line.replace("- ", ""),
      });
    }
    if (line.startsWith("•")) {
      blocks.push({
        type: "exercise",
        text: line.replace("• ", ""),
      });
    }
  }

  return blocks;
}