import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { 
  generateChatResponse, 
  generateWorkout, 
  generateScript, 
  generateReflection 
} from "./openai";
import { 
  insertChatMessageSchema, 
  insertWorkoutSchema, 
  insertScriptSchema, 
  insertReflectionSchema 
} from "@shared/schema";
import { randomUUID } from "crypto";

export async function registerRoutes(app: Express): Promise<Server> {
  // Chat routes
  app.get("/api/chat", async (req, res) => {
    try {
      const messages = await storage.getChatMessages();
      res.json(messages);
    } catch (error) {
      console.error("Error fetching chat messages:", error);
      res.status(500).json({ error: "Failed to fetch messages" });
    }
  });

  app.post("/api/chat", async (req, res) => {
    try {
      const validatedData = insertChatMessageSchema.parse(req.body);
      
      // Store user message
      const userMessage = await storage.addChatMessage(validatedData);
      
      // Generate AI response
      const aiResponse = await generateChatResponse(validatedData.content);
      
      // Store AI message
      const assistantMessage = await storage.addChatMessage({
        role: "assistant",
        content: aiResponse,
      });
      
      // Return both messages
      res.json({ userMessage, assistantMessage });
    } catch (error: any) {
      console.error("Error in chat:", error);
      if (error.name === "ZodError") {
        res.status(400).json({ error: "Invalid request data" });
      } else {
        res.status(500).json({ error: "Failed to process chat message" });
      }
    }
  });

  // Workout routes
  app.get("/api/workouts", async (req, res) => {
    try {
      const workouts = await storage.getWorkouts();
      res.json(workouts);
    } catch (error) {
      console.error("Error fetching workouts:", error);
      res.status(500).json({ error: "Failed to fetch workouts" });
    }
  });

  app.post("/api/workouts/generate", async (req, res) => {
    try {
      const validatedData = insertWorkoutSchema.parse(req.body);
      
      const generatedWorkout = await generateWorkout(
        validatedData.goal,
        validatedData.duration,
        validatedData.equipment || "",
        validatedData.fitnessLevel
      );
      
      const workout = {
        id: randomUUID(),
        title: generatedWorkout.title,
        exercises: generatedWorkout.exercises,
        timestamp: new Date().toISOString(),
      };
      
      await storage.addWorkout(workout);
      res.json(workout);
    } catch (error: any) {
      console.error("Error generating workout:", error);
      if (error.name === "ZodError") {
        res.status(400).json({ error: "Invalid request data" });
      } else {
        res.status(500).json({ error: "Failed to generate workout" });
      }
    }
  });

  // Script routes
  app.get("/api/scripts", async (req, res) => {
    try {
      const scripts = await storage.getScripts();
      res.json(scripts);
    } catch (error) {
      console.error("Error fetching scripts:", error);
      res.status(500).json({ error: "Failed to fetch scripts" });
    }
  });

  app.post("/api/scripts/generate", async (req, res) => {
    try {
      const validatedData = insertScriptSchema.parse(req.body);
      
      const generatedScript = await generateScript(
        validatedData.topic,
        validatedData.scriptType,
        validatedData.duration,
        validatedData.tone
      );
      
      const script = {
        id: randomUUID(),
        title: generatedScript.title,
        type: generatedScript.type,
        content: generatedScript.content,
        timestamp: new Date().toISOString(),
      };
      
      await storage.addScript(script);
      res.json(script);
    } catch (error: any) {
      console.error("Error generating script:", error);
      if (error.name === "ZodError") {
        res.status(400).json({ error: "Invalid request data" });
      } else {
        res.status(500).json({ error: "Failed to generate script" });
      }
    }
  });

  // Reflection routes
  app.get("/api/reflections", async (req, res) => {
    try {
      const reflections = await storage.getReflections();
      res.json(reflections);
    } catch (error) {
      console.error("Error fetching reflections:", error);
      res.status(500).json({ error: "Failed to fetch reflections" });
    }
  });

  app.post("/api/reflections/generate", async (req, res) => {
    try {
      const validatedData = insertReflectionSchema.parse(req.body);
      
      const generatedReflection = await generateReflection(
        validatedData.period,
        validatedData.focus
      );
      
      const reflection = {
        id: randomUUID(),
        title: generatedReflection.title,
        wins: generatedReflection.wins,
        challenges: generatedReflection.challenges,
        growth: generatedReflection.growth,
        tags: generatedReflection.tags,
        timestamp: new Date().toISOString(),
      };
      
      await storage.addReflection(reflection);
      res.json(reflection);
    } catch (error: any) {
      console.error("Error generating reflection:", error);
      if (error.name === "ZodError") {
        res.status(400).json({ error: "Invalid request data" });
      } else {
        res.status(500).json({ error: "Failed to generate reflection" });
      }
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
