// backend/routes/resume.js
import { loadMemory } from "../engine/memory.js";
import { buildECM } from "../engine/intentEngine.js";

export default async function resumeHandler(req, res) {
  const userId = req.query.id;
  if (!userId) return res.status(400).json({ ok: false });

  const memory = await loadMemory(userId);
  const last = memory.lastECM;
  if (!last) return res.json({ ok: true, show: false });

  return res.json({
    ok: true,
    show: true,
    cue: last.resumeCue,
    intent: last.intent,
    confidence: last.confidence,
  });
}
