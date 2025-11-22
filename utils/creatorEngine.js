// utils/creatorEngine.js — Spirit v5.1 Showcase Creator Engine

export function buildCreatorContent({ topic, platform, identity, tone }) {
  const hook = `Here's the truth: ${topic
    .charAt(0)
    .toUpperCase() + topic.slice(1)} is holding people back — until today.`;

  const shortScript = `
    Hook: "${hook}"
    Core Insight: Give one bold truth.
    Action: Give them one practical takeaway they can do within 60 seconds.
  `;

  const longScript = `
    Title: "${topic}"
    - Begin with tension or a personal insight
    - Break the topic into 3 power sections
    - End with a transformation the audience imagines themselves achieving
  `;

  const caption = `🔥 ${topic}\nYou’re closer than you think.\n— Spirit`;

  return {
    platform: platform || "tiktok",
    hook,
    shortScript,
    longScript,
    caption,
    identity: identity || "default",
    tone: tone || "clear"
  };
}
