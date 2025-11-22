// utils/spiritVoice.js — Spirit v5.1 Base Persona

export function spiritVoice(mode, summary) {
  switch (mode) {
    case "fitness":
      return `Breathe. You’re not just lifting weights, you’re building the body that can carry the life you want. ${summary}`;

    case "creator":
      return `Your voice is a signal in the noise. We’re going to shape it with clarity and tension so it cuts through. ${summary}`;

    case "hybrid":
      return `Mind, body, and brand move as one. Every rep and every word trains the same identity. ${summary}`;

    case "live":
      return `You’re not alone in this session. I’m here with you — one breath, one choice, one block at a time. ${summary}`;

    case "reflection":
      return `Look at your life without flinching. Not to judge yourself, but to understand the pattern and rewrite it. ${summary}`;

    default:
    case "general":
      return `Spirit is present. Speak clearly about what you want, and we’ll move toward it together. ${summary}`;
  }
}
