import { runSpirit } from "../engine/executor.js";

/* ---------- ephemeral session store ---------- */
const liveSessions = new Map();

function getLiveSession(id) {
  if (!liveSessions.has(id)) {
    liveSessions.set(id, {
      id,
      state: "IDLE",
      setIndex: 0,
      failures: 0,
      lastRPE: null,
    });
  }
  return liveSessions.get(id);
}

/* ---------- entry valve ---------- */
export async function liveController(req, res) {
  const { sessionId, event, RPE, symptoms } = req.body;
  if (!sessionId) return res.status(400).json({ ok: false, error: "sessionId required" });

  const session = getLiveSession(sessionId);

  switch (session.state) {
    case "IDLE":
      return startSession(session, res);
    case "SET_IN_PROGRESS":
      return res.status(429).json({ ok: false, error: "set not finished" });
    case "SET_REPORTED":
      return processReport(session, { event, RPE, symptoms }, res);
    case "SET_READY":
      return dispatchNextSet(session, res);
    case "TERMINATED":
      return res.json({ ok: true, intent: "ended" });
    default:
      return res.status(500).json({ ok: false, error: "unknown state" });
  }
}

/* ---------- hard reset ---------- */
async function startSession(session, res) {
  session.state = "SET_READY";
  session.setIndex = 0;
  session.failures = 0;
  session.lastRPE = null;
  return dispatchNextSet(session, res);
}

/* ---------- permission gate ---------- */
async function dispatchNextSet(session, res) {
  session.state = "SET_IN_PROGRESS";

  /* ask Spirit for ONE set only */
  const spiritRes = await runSpirit({
    userId: session.id,
    message: `Live set ${session.setIndex + 1}, last RPE ${session.lastRPE ?? "none"}.`,
    mode: "live",
    taskType: "live_set",
  });

  if (!spiritRes.ok) {
    session.state = "TERMINATED";
    return res.status(500).json({ ok: false, error: "Spirit failed" });
  }

  const { reply } = spiritRes;
  const lines = reply.split("\n").filter(Boolean);
  const instruction = lines[0]?.replace(/^[-•]\s*/, "") ?? "";
  const cue = lines[1]?.replace(/^[-•]\s*/, "") ?? null;

  return res.json({
    ok: true,
    intent: "live_set",
    say: instruction,
    cue,
    awaits: ["completed", "failed", "stop"],
  });
}

/* ---------- user report ---------- */
function processReport(session, report, res) {
  const { event, RPE, symptoms } = report;

  if (event === "stop") return terminate(session, res);
  if (symptoms?.length) return safetyOverride(session, res);

  session.lastRPE = RPE ?? session.lastRPE;
  if (event === "failed") session.failures += 1;

  if (session.failures >= 2 || (session.lastRPE && session.lastRPE >= 9)) {
    return terminate(session, res);
  }

  session.setIndex += 1;
  session.state = "SET_READY";
  return dispatchNextSet(session, res);
}

/* ---------- safety override ---------- */
function safetyOverride(session, res) {
  session.state = "TERMINATED";
  liveSessions.delete(session.id);
  return res.json({ ok: true, intent: "ended", say: "Session ended for safety." });
}

/* ---------- termination + write-back ---------- */
function terminate(session, res) {
  session.state = "TERMINATED";
  /* push ultra-short summary to fitness memory */
  fetch(`/spirit/memory`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      userId: session.id,
      liveWorkoutSummary: {
        setsCompleted: session.setIndex,
        avgRPE: session.lastRPE,
        failures: session.failures,
      },
    }),
  }).catch(() => {}); // ignore network errors
  liveSessions.delete(session.id);
  return res.json({ ok: true, intent: "ended" });
}
