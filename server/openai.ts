import OpenAI from "openai";

// This is using Replit's AI Integrations service, which provides OpenAI-compatible API access without requiring your own OpenAI API key.
// From the blueprint: javascript_openai_ai_integrations
const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

export async function generateChatResponse(userMessage: string): Promise<string> {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-5.1", // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
      messages: [
        {
          role: "system",
          content: "You are Spirit, a friendly and helpful AI assistant. You provide thoughtful, concise, and helpful responses.",
        },
        {
          role: "user",
          content: userMessage,
        },
      ],
      max_completion_tokens: 8192,
    });
    return response.choices[0]?.message?.content || "I apologize, but I couldn't generate a response.";
  } catch (error) {
    console.error("Error generating chat response:", error);
    throw new Error("Failed to generate response");
  }
}

export async function generateWorkout(goal: string, duration: string, equipment: string, fitnessLevel: string): Promise<{ title: string; exercises: any[] }> {
  try {
    const prompt = `Generate a personalized workout plan with the following details:
- Goal: ${goal}
- Duration: ${duration} minutes
- Equipment: ${equipment || "No equipment (bodyweight)"}
- Fitness Level: ${fitnessLevel}

Return a JSON object with:
- title: A motivating title for the workout
- exercises: An array of exercises, each with:
  - name: Exercise name
  - sets: Number of sets (if applicable)
  - reps: Number of reps or rep range (if applicable)
  - duration: Duration (if time-based exercise)
  - notes: Brief form tips or modifications

Make it practical and achievable.`;

    const response = await openai.chat.completions.create({
      model: "gpt-5.1", // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      max_completion_tokens: 8192,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      throw new Error("No response content");
    }

    return JSON.parse(content);
  } catch (error) {
    console.error("Error generating workout:", error);
    throw new Error("Failed to generate workout");
  }
}

export async function generateScript(topic: string, scriptType: string, duration: string | undefined, tone: string): Promise<{ title: string; type: string; content: string }> {
  try {
    const durationText = duration ? ` for approximately ${duration}` : "";
    const prompt = `Generate a ${scriptType} script${durationText} on the following topic:
Topic: ${topic}
Tone: ${tone}

Return a JSON object with:
- title: A catchy title for the script
- type: The type of script (${scriptType})
- content: The full script content with appropriate formatting, sections, and flow

Make it engaging, well-structured, and ready to use.`;

    const response = await openai.chat.completions.create({
      model: "gpt-5.1", // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      max_completion_tokens: 8192,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      throw new Error("No response content");
    }

    return JSON.parse(content);
  } catch (error) {
    console.error("Error generating script:", error);
    throw new Error("Failed to generate script");
  }
}

export async function generateReflection(period: string, focus: string | undefined): Promise<{ title: string; wins: string[]; challenges: string[]; growth: string[]; tags: string[] }> {
  try {
    const focusText = focus ? ` focusing on ${focus}` : "";
    const prompt = `Generate a guided reflection for ${period}${focusText}.

Return a JSON object with:
- title: A reflective title for this period
- wins: Array of 3-5 wins or achievements
- challenges: Array of 3-5 challenges or obstacles faced
- growth: Array of 3-5 growth areas or lessons learned
- tags: Array of 3-5 relevant theme tags

Make it thoughtful, specific, and encouraging.`;

    const response = await openai.chat.completions.create({
      model: "gpt-5.1", // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      max_completion_tokens: 8192,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      throw new Error("No response content");
    }

    return JSON.parse(content);
  } catch (error) {
    console.error("Error generating reflection:", error);
    throw new Error("Failed to generate reflection");
  }
}
