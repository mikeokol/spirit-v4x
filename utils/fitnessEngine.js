// utils/fitnessEngine.js — Spirit v5.1 Showcase Fitness Engine

export function buildFitnessBlock({ goal, experience, days, gender, height, weight }) {
  const base = {
    hypertrophy: ["Push", "Pull", "Legs", "Upper", "Lower"],
    strength: ["Full Body Strength", "Lower Strength", "Upper Strength"],
    "fat loss": ["Circuits", "Cardio + Weights", "Mixed Conditioning"],
    endurance: ["Engine Work", "Zone 2", "Intervals"],
    recomposition: ["Full Body", "Upper/Lower", "Accessory + Core"]
  };

  const structure = base[goal.toLowerCase()] || ["Full Body", "Accessory", "Core"];

  const weeklyStructure = Array.from({ length: days }).map((_, idx) => {
    return {
      day: idx + 1,
      focus: structure[idx % structure.length]
    };
  });

  const focusPoints = [
    "Prioritize form before intensity.",
    "Increase weight only when reps feel stable.",
    "Control tempo — especially the lowering phase."
  ];

  const identityAnchor = `Become the version of you that finishes the plan, not just starts it.`;

  return {
    goal,
    experience,
    days,
    weeklyStructure,
    focusPoints,
    identityAnchor,
    userMetrics: { gender, height, weight }
  };
}
