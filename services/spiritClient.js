// services/spiritClient.js — Unified Prompt Caller for Spirit v5.1

import OpenAI from "openai";
import "dotenv/config";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

export const spiritClient = {
  callModel: async (systemPrompt, modePrompt, userPayload) => {
    const fullPrompt = `
${systemPrompt}

${modePrompt}
    `;

    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.4,
      messages: [
        { role: "system", content: fullPrompt },
        { role: "user", content: JSON.stringify(userPayload) }
      ]
    });

    const text = completion.choices[0].message.content;

    try {
      return JSON.parse(text);
    } catch (err) {
      throw new Error("Invalid JSON from model:\n" + text);
    }
  }
};