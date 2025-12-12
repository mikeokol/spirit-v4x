// backend/engine/clearECM.js
// Removes last resume anchor – no UI, no drama

import { loadMemory, saveMemory } from "./memory.js";

export async function clearECM(userId) {
  const memory = await loadMemory(userId);
  delete memory.lastECM;
  await saveMemory(userId, memory);
}
