// engine/parser.js
// Spirit v7 — Workout Plan Parser
// Converts Spirit's raw fitness text into structured blocks usable by Live Mode.

export function parseWorkoutPlan(raw) {
  if (!raw || typeof raw !== "string") {
    return { warmup: null, blocks: [], cooldown: null };
  }

  const lines = raw.split("\n").map(l => l.trim());
  let warmup = null;
  let cooldown = null;
  let blocks = [];
  let currentBlock = null;

  for (let line of lines) {
    // -------------------------------
    // Detect warmup section
    // -------------------------------
    if (/^warm[- ]?up/i.test(line)) {
      warmup = extractExercises(line);
      continue;
    }

    // -------------------------------
    // Detect cooldown section
    // -------------------------------
    if (/^cool[- ]?down/i.test(line)) {
      cooldown = extractExercises(line);
      continue;
    }

    // -------------------------------
    // Detect block or circuit header
    // -------------------------------
    if (/^(block|circuit|round|main)[^\:]*\:/i.test(line)) {
      if (currentBlock) {
        blocks.push(currentBlock);
      }
      currentBlock = {
        title: line.replace(/\:$/, ""),
        exercises: [],
      };
      continue;
    }

    // -------------------------------
    // Detect exercises inside a block
    // -------------------------------
    if (currentBlock && /^[-•0-9]/.test(line)) {
      currentBlock.exercises.push(parseExercise(line));
      continue;
    }
  }

  if (currentBlock) {
    blocks.push(currentBlock);
  }

  return {
    warmup,
    blocks,
    cooldown,
  };
}

// =========================================================
// HELPERS
// =========================================================
function extractExercises(line) {
  return line.replace(/^.*?:/, "").trim();
}

function parseExercise(line) {
  const base = line.replace(/^[-•]/, "").trim();

  // Extract repetitions like "3x10"
  const match = base.match(/(\d+)\s*x\s*(\d+)/i);

  return {
    raw: base,
    sets: match ? Number(match[1]) : null,
    reps: match ? Number(match[2]) : null,
  };
}