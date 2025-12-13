// engine/fitnessParser.js
// Spirit v7 — Fitness Response Parser (loose, natural-language safe)

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

  /* -------- macros (case-insensitive, colon optional) -------- */
  const cal  = text.match(/(\d+)\s*(calories?|kcal)/i);
  const prot = text.match(/(\d+)\s*g?\s*protein/i);
  const carb = text.match(/(\d+)\s*g?\s*carbs?/i);
  const fat  = text.match(/(\d+)\s*g?\s*fats?/i);
  const h2o  = text.match(/(\d+(?:\.\d+)?)\s*L?\s*(water|litres?|liters?)/i);
  if (cal)  data.calories   = parseInt(cal[1],  10);
  if (prot) data.protein    = parseInt(prot[1], 10);
  if (carb) data.carbs      = parseInt(carb[1], 10);
  if (fat)  data.fats       = parseInt(fat[1],  10);
  if (h2o)  data.waterLitres= parseFloat(h2o[1]);

  /* -------- weekly split (grab every day block until empty line) -------- */
  const dayBlock = text.match(/(monday|tuesday|wednesday|thursday|friday|saturday|sunday)[:\s\-]+([^\n]+)/gi);
  if (dayBlock) data.weeklySplit = dayBlock.map(l => l.trim());

  /* -------- meals (any line with kcal / calories) -------- */
  const mealLines = text.match(/^.*\d+\s*(kcal|calories?).*$/gim);
  if (mealLines) {
    mealLines.forEach(l => {
      const cals = l.match(/(\d+)\s*(kcal|calories?)/i);
      if (cals) data.meals.push({ time: "Flexible", food: l.replace(/^[-•]\s*/,"").trim(), cals: parseInt(cals[1],10) });
    });
  }

  /*  -------- supplements (bullet or keyword) -------- */
  const suppLines = text.split('\n').filter(l =>
    l.match(/^[-•]\s*(creatine|protein|multivitamin|omega|vitamin|bcaa|zinc|magnesium)/i) ||
    l.toLowerCase().includes('supplement')
  );
  suppLines.forEach(l => data.supplements.push(l.replace(/^[-•]\s*/,"").trim()));

  return data;
}
