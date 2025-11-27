// services/openai.js
import OpenAI from "openai";

let client = null;

export function getClient() {
  if (client) return client;

  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    throw new Error("Missing OPENAI_API_KEY environment variable.");
  }

  client = new OpenAI({ apiKey });
  return client;
}