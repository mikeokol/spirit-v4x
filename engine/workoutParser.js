// engine/workoutParser.js
export function parseWorkoutPlan(planText = "") {
  const data = {
    calories: 0,
    protein: 0,
    carbs: 0,
    fats: 0,
    waterLitres: 0,
    weeklySplit: [],
    meals: [],
    supplements: []
  };

  const calorieMatch = planText.match(/(\d+)\s*(calories?|kcal)/i);
  const proteinMatch = planText.match(/(\d+)\s*g?\s*protein/i);
  const carbsMatch = planText.match(/(\d+)\s*g?\s*carbs/i);
  const fatsMatch = planText.match(/(\d+)\s*g?\s*fats/i);
  const waterMatch = planText.match(/(\d+(?:\.\d+)?)\s*L?\s*(water|litres?|liters?)/i);

  if (calorieMatch) data.calories = parseInt(calorieMatch[1], 10);
  if (proteinMatch) data.protein = parseInt(proteinMatch[1], 10);
  if (carbsMatch) data.carbs = parseInt(carbsMatch[1], 10);
  if (fatsMatch) data.fats = parseInt(fatsMatch[1], 10);
  if (waterMatch) data.waterLitres = parseFloat(waterMatch[1]);

  const days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];
  days.forEach(day => {
    const dayRegex = new RegExp(`${day}[:\\s-]+([^\n]+)`, 'gi');
    const match = planText.match(dayRegex);
    if (match) {
      data.weeklySplit.push({
        day: day.charAt(0).toUpperCase() + day.slice(1),
        exercises: match[1].split(',').map(ex => ex.trim())
      });
    }
  });

  const workoutRegex = /(\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*(\d+)\s*sets?\s*x\s*(\d+)\s*reps?/g;
  let match;
  while ((match = workoutRegex.exec(planText)) !== null) {
    const [, weight, exercise, sets, reps] = match;
    data.weeklySplit.forEach(day => {
      day.exercises.forEach(exerciseItem => {
        if (exerciseItem.includes(exercise)) {
          exerciseItem += ` - ${sets} sets x ${reps} reps`;
        }
      });
    });
  }

  const mealMatches = planText.match(/(\d+(?:\.\d+)?)\s*kc?al/gi);
  if (mealMatches) {
    const lines = planText.split('\n');
    lines.forEach(line => {
      if (line.toLowerCase().includes('meal') && line.match(/\d+\s*kc?al/i)) {
        const calories = line.match(/(\d+)\s*kc?al/i);
        if (calories) {
          data.meals.push({
            time: "Flexible",
            food: line.replace(/[-•]\s*/, "").trim(),
            cal: parseInt(calories[1], 10)
          });
        }
      }
    });
  }

  const supplementLines = planText.split('\n').filter(line => 
    line.toLowerCase().includes('supplement') || 
    line.match(/^[-•]\s*(creatine|protein|multivitamin|omega|vitamin)/i)
  );
  
  supplementLines.forEach(line => {
    const cleanLine = line.replace(/^[-•]\s*/, "").trim();
    if (cleanLine) data.supplements.push(cleanLine);
  });

  return data;
}
