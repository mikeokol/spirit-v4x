import { z } from "zod";

// Chat Message Schema
export const chatMessageSchema = z.object({
  id: z.string(),
  role: z.enum(["user", "assistant"]),
  content: z.string(),
  timestamp: z.string(),
});

export const insertChatMessageSchema = chatMessageSchema.omit({ id: true, timestamp: true });

export type ChatMessage = z.infer<typeof chatMessageSchema>;
export type InsertChatMessage = z.infer<typeof insertChatMessageSchema>;

// Workout Schema
export const workoutSchema = z.object({
  id: z.string(),
  title: z.string(),
  exercises: z.array(z.object({
    name: z.string(),
    sets: z.number().optional(),
    reps: z.string().optional(),
    duration: z.string().optional(),
    notes: z.string().optional(),
  })),
  timestamp: z.string(),
});

export const insertWorkoutSchema = z.object({
  goal: z.string().min(3, "Goal must be at least 3 characters"),
  duration: z.string().min(1, "Duration is required"),
  equipment: z.string().optional(),
  fitnessLevel: z.enum(["beginner", "intermediate", "advanced"]).default("intermediate"),
});

export type Workout = z.infer<typeof workoutSchema>;
export type InsertWorkout = z.infer<typeof insertWorkoutSchema>;

// Script Schema
export const scriptSchema = z.object({
  id: z.string(),
  title: z.string(),
  type: z.string(),
  content: z.string(),
  timestamp: z.string(),
});

export const insertScriptSchema = z.object({
  topic: z.string().min(3, "Topic must be at least 3 characters"),
  scriptType: z.enum(["video", "podcast", "presentation", "social"]).default("video"),
  duration: z.string().optional(),
  tone: z.enum(["professional", "casual", "educational", "entertaining"]).default("professional"),
});

export type Script = z.infer<typeof scriptSchema>;
export type InsertScript = z.infer<typeof insertScriptSchema>;

// Reflection Schema
export const reflectionSchema = z.object({
  id: z.string(),
  title: z.string(),
  wins: z.array(z.string()),
  challenges: z.array(z.string()),
  growth: z.array(z.string()),
  tags: z.array(z.string()),
  timestamp: z.string(),
});

export const insertReflectionSchema = z.object({
  period: z.string().min(3, "Period must be at least 3 characters"),
  focus: z.string().optional(),
});

export type Reflection = z.infer<typeof reflectionSchema>;
export type InsertReflection = z.infer<typeof insertReflectionSchema>;
