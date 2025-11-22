// utils/classifyIntent.js — Spirit v5.1 Showcase Intent Detector

export const classifyIntent = (message = "") => {
  const lower = message.toLowerCase();

  const fitnessKeywords = [
    "workout","training","gym","exercise","lift",
    "leg day","push day","pull day","upper","lower",
    "program","routine","fat loss","lose weight"
  ];

  const creatorKeywords = [
    "script","content","video","hook","reels",
    "tiktok","youtube","caption","edit","viral"
  ];

  const reflectionKeywords = [
    "i feel","i am feeling","i don't know",
    "identity","discipline","motivation",
    "lost","why","intention","what should i do"
  ];

  const liveKeywords = [
    "coach me","walk me through","guide me",
    "live","session now","step by step"
  ];

  const fitness = fitnessKeywords.some(k => lower.includes(k));
  const creator = creatorKeywords.some(k => lower.includes(k));
  const reflection = reflectionKeywords.some(k => lower.includes(k));
  const live = liveKeywords.some(k => lower.includes(k));

  if (live) return "live";
  if (fitness && creator) return "hybrid";
  if (fitness) return "fitness";
  if (creator) return "creator";
  if (reflection) return "reflection";

  return "general";
};
