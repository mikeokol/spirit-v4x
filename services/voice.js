// services/voice.js — Spirit v7.2 Voice Engine

import { getClient } from "./openai.js";

export async function speak(text) {
  try {
    const client = getClient();

    const response = await client.audio.speech.create({
      model: "gpt-4o-audio-preview",
      voice: "alloy",
      format: "mp3",
      input: text,
    });

    // Returned as a base64 MP3
    const audioBase64 = response.data.audio;
    const audioUrl = `data:audio/mp3;base64,${audioBase64}`;

    return { ok: true, audioUrl };
  } catch (err) {
    console.error("[voice] TTS error:", err);
    return { ok: false, audioUrl: null };
  }
}

/**
 * Auto-coached pacing: Spirit says:
 * "Nice. Take 30 seconds rest. I’ll tell you when to start again."
 * Then provides countdown audio (optional)
 */
export async function speakRest(seconds = 30) {
  const text = `Nice work. Take ${seconds} seconds rest. I will let you know when to start again.`;
  return speak(text);
}

/**
 * Motivational cues (short)
 */
export function getMotivationCue(intensity) {
  if (intensity <= 4) {
    return "Good warmup pace. Keep breathing steady.";
  }
  if (intensity <= 7) {
    return "Strong work — you're right where you need to be.";
  }
  return "Power through — stay controlled, you're doing great.";
}