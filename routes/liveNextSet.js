import { loadUserMemory, saveUserMemory } from "../engine/memory.js";
import { runSpirit } from "../engine/executor.js";

export default async function liveNextSetHandler(req, res) {
  const { userId, markDone = false, chat = "" } = req.body;
  if (!userId) return res.status(400).json({ ok: false, error: "Missing userId" });

  const memory = await loadUserMemory(userId);

  /* first call – store the list if not present */
  if (!memory.liveWorkoutList) {
    const raw = memory.lastWorkoutRaw || "";
    const list = parseList(raw);
    memory.liveWorkoutList = list;
    memory.liveIndex = 0;
    await saveUserMemory(userId, memory);
  }

  const list = memory.liveWorkoutList;
  let idx = memory.liveIndex || 0;

  /* user chatted – answer but keep index */
  if (chat) {
    const answer = await coachChat(chat, list[idx]);
    return res.json({ ok: true, reply: answer, continueIndex: idx });
  }

  /* mark done – advance */
  if (markDone) idx += 1;
  if (idx >= list.length) {
    await saveUserMemory(userId, { ...memory, liveWorkoutList: null, liveIndex: 0 });
    return res.json({ ok: true, finished: true, reply: "Great session! Hydrate and rest. See you next time." });
  }

  memory.liveIndex = idx;
  await saveUserMemory(userId, memory);
  const next = list[idx];
  const cue = await coachCue(next);
  return res.json({ ok: true, exercise: next, reply: cue });

  /* ---------- helpers ---------- */
  function parseList(text) {
    return text
      .split("\n")
      .filter((l) => /^[-•]/.test(l.trim()))
      .map((l) => {
        const name = l.replace(/^[-•]\s*/, "").split(/ \(/)[0].trim();
        const sets = Number(l.match(/(\d+)\s*sets?/i)?.[1] || 3);
        const reps = Number(l.match(/(\d+)\s*reps?/i)?.[1] || 10);
        return { name, sets, reps };
      });
  }

  async function coachCue(ex) {
    const res = await runSpirit({
      userId,
      message: `Concise cue for ${ex.sets}×${ex.reps} ${ex.name}. Keep it short, motivational, present tense.`,
      mode: "fitness",
      taskType: "live_coaching",
    });
    return res.reply || `Next: ${ex.sets}×${ex.reps} ${ex.name}`;
  }

  async function coachChat(text, currentEx) {
    const res = await runSpirit({
      userId,
      message: text,
      mode: "fitness",
      taskType: "live_coaching",
      meta: { currentExercise: currentEx },
    });
    return res.reply || "Got it. Ready to continue?";
  }
}
