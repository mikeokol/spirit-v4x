// classifier.js — Lightweight Intent Detector for Spirit v5.1

export const classifyIntent = (message) => {
  const lower = message.toLowerCase();

  // FITNESS
  if (
    lower.includes("workout") ||
    lower.includes("training") ||
    lower.includes("gym") ||
    lower.includes("exercise") ||
    lower.includes("leg day") ||
    lower.includes("upper") ||
    lower.includes("session")
  ) {
    return "fitness";
  }

  // CREATOR
  if (
    lower.includes("script") ||
    lower.includes("content") ||
    lower.includes("video") ||
    lower.includes("hook") ||
    lower.includes("reels") ||
    lower.includes("tiktok") ||
    lower.includes("youtube")
  ) {
    return "creator";
  }

  // REFLECTION
  if (
    lower.includes("i feel") ||
    lower.includes("i am feeling") ||
    lower.includes("intention") ||
    lower.includes("why") ||
    lower.includes("identity") ||
    lower.includes("lost") ||
    lower.includes("discipline")
  ) {
    return "reflection";
  }

  return "general";
};