// engine/fitnessParser.js
// Spirit v7 — Fitness Response Parser
// Extracts structured macros, meals, supplements from AI response

export function parseFitnessResponse(text = "") {
  if (!text) return null;

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

  // Extract macros - multiple formats
  const calorieMatch = text.match(/(\d+)\s*(calories?|kcal)/i);
  const proteinMatch = text.match(/(\d+)\s*g?\s*protein/i);
  const carbsMatch = text.match(/(\d+)\s*g?\s*carbs/i);
  const fatsMatch = text.match(/(\d+)\s*g?\s*fats/i);
  const waterMatch = text.match(/(\d+(?:\.\d+)?)\s*L?\s*(water|litres?|liters?)/i);

  if (calorieMatch) data.calories = parseInt(calorieMatch[1]);
  if (proteinMatch) data.protein = parseInt(proteinMatch[1]);
  if (carbsMatch) data.carbs = parseInt(carbsMatch[1]);
  if (fatsMatch) data.fats = parseInt(fatsMatch[1]);
  if (waterMatch) data.waterLitres = parseFloat(waterMatch[1]);

  // Extract weekly split
  const days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];
  days.forEach(day => {
    const dayRegex = new RegExp(`${day}[:\\s-]+([^\n]+)`, 'gi');
    const match = text.match(dayRegex);
    if (match) {
      data.weeklySplit.push(match[0].trim());
    }
  });

  // Extract meals (basic pattern)
  const mealMatches = text.match(/(\d+(?:\.\d+)?)\s*kc?al/gi);
  if (mealMatches) {
    // Simple meal extraction - you can enhance this
    const lines = text.split('\n');
    lines.forEach(line => {
      if (line.toLowerCase().includes('meal') && line.match(/\d+\s*kc?al/i)) {
        const calories = line.match(/(\d+)\s*kc?al/i);
        if (calories) {
          data.meals.push({
            time: "Flexible",
            food: line.replace(/[-•]\s*/, "").trim(),
            cals: parseInt(calories[1])
          });
        }
      }
    });
  }

  // Extract supplements
  const supplementLines = text.split('\n').filter(line => 
    line.toLowerCase().includes('supplement') || 
    line.match(/^[-•]\s*(creatine|protein|multivitamin|omega|vitamin)/i)
  );
  
  supplementLines.forEach(line => {
    const cleanLine = line.replace(/^[-•]\s*/, "").trim();
    if (cleanLine) data.supplements.push(cleanLine);
  });

  return data;
}
