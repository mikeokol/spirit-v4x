// backend/engine/intentEngine.js
// Micro-engine for ECM + Confidence + Intent + PRP

export function classifyIntent(text) {
  const lower = text.toLowerCase();
  if (/feel|sad|angry|overwhelmed|hurt/i.test(lower)) return { type: "venting", confidence: 0.9 };
  if (/should I|decide|pick|choose|option/i.test(lower)) return { type: "decision", confidence: 0.8 };
  if (/plan|schedule|steps|next week/i.test(lower)) return { type: "planning", confidence: 0.8 };
  if (/why|what if|meaning|purpose/i.test(lower)) return { type: "exploratory", confidence: 0.7 };
  if (/reflect|learned|realised|grateful/i.test(lower)) return { type: "reflection", confidence: 0.9 };
  return { type: "general", confidence: 0.5 };
}

export function scoreConfidence(userId, text, meta = {}) {
  if (meta.directStatement) return 1.0;
  if (meta.inferred) return 0.6;
  if (meta.pattern) return 0.3;
  return 0.5;
}

export function buildECM(userId, intent, stopReason) {
  return {
    userId,
    ts: Date.now(),
    intent,
    stopReason,
    resumeCue: `Last time, you were ${intent.type === "venting" ? "processing emotions" : intent.type}. Want to continue or switch?`,
    confidence: intent.confidence,
  };
}
