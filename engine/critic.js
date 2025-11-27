// engine/critic.js — Spirit v7.1 Critic (lightweight, offline)

export async function runCritic({ reply, plan, mode, taskType }) {
  const length_ok = typeof reply === "string" && reply.length < 2400;

  return {
    ok: true,
    notes: {
      length_ok,
      has_plan: !!plan,
      mode: mode || null,
      taskType: taskType || null,
      aligned: true, // we keep this simple for now
    },
  };
}