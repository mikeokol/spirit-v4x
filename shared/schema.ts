import { z } from "zod";
import { pgTable, text, varchar, timestamp, jsonb } from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";

// Drizzle Tables with database-generated defaults
export const chatMessages = pgTable("chat_messages", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  role: varchar("role", { length: 20 }).notNull(),
  content: text("content").notNull(),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
});

export const workouts = pgTable("workouts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  title: text("title").notNull(),
  exercises: jsonb("exercises").notNull(),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
});

export const scripts = pgTable("scripts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  title: text("title").notNull(),
  type: varchar("type", { length: 50 }).notNull(),
  content: text("content").notNull(),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
});

export const reflections = pgTable("reflections", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  title: text("title").notNull(),
  wins: jsonb("wins").notNull(),
  challenges: jsonb("challenges").notNull(),
  growth: jsonb("growth").notNull(),
  tags: jsonb("tags").notNull(),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
});

// Exercise schema for validation
const exerciseSchema = z.object({
  name: z.string(),
  sets: z.number().nullable().optional(),
  reps: z.string().nullable().optional(),
  duration: z.string().nullable().optional(),
  notes: z.string().nullable().optional(),
});

// Domain types with ISO string timestamps (for API/frontend compatibility)
export type ChatMessage = {
  id: string;
  role: string;
  content: string;
  timestamp: string;
};

export type Workout = {
  id: string;
  title: string;
  exercises: Array<{
    name: string;
    sets?: number;
    reps?: string;
    duration?: string;
    notes?: string;
  }>;
  timestamp: string;
};

export type Script = {
  id: string;
  title: string;
  type: string;
  content: string;
  timestamp: string;
};

export type Reflection = {
  id: string;
  title: string;
  wins: string[];
  challenges: string[];
  growth: string[];
  tags: string[];
  timestamp: string;
};

// Insert schemas (for database inserts, omitting generated fields)
export const insertChatMessageSchema = createInsertSchema(chatMessages, {
  role: z.enum(["user", "assistant"]),
}).omit({
  id: true,
  timestamp: true,
});

export const insertWorkoutDBSchema = createInsertSchema(workouts, {
  exercises: z.array(exerciseSchema),
}).omit({
  id: true,
  timestamp: true,
});

export const insertScriptDBSchema = createInsertSchema(scripts).omit({
  id: true,
  timestamp: true,
});

export const insertReflectionDBSchema = createInsertSchema(reflections, {
  wins: z.array(z.string()),
  challenges: z.array(z.string()),
  growth: z.array(z.string()),
  tags: z.array(z.string()),
}).omit({
  id: true,
  timestamp: true,
});

// Form validation schemas (for generation requests)
export const insertWorkoutSchema = z.object({
  goal: z.string().min(3, "Goal must be at least 3 characters"),
  duration: z.string().min(1, "Duration is required"),
  equipment: z.string().optional(),
  fitnessLevel: z.enum(["beginner", "intermediate", "advanced"]).default("intermediate"),
});

export const insertScriptSchema = z.object({
  topic: z.string().min(3, "Topic must be at least 3 characters"),
  scriptType: z.enum(["video", "podcast", "presentation", "social"]).default("video"),
  duration: z.string().optional(),
  tone: z.enum(["professional", "casual", "educational", "entertaining"]).default("professional"),
});

export const insertReflectionSchema = z.object({
  period: z.string().min(3, "Period must be at least 3 characters"),
  focus: z.string().optional(),
});

export type InsertChatMessage = z.infer<typeof insertChatMessageSchema>;
export type InsertWorkout = z.infer<typeof insertWorkoutSchema>;
export type InsertScript = z.infer<typeof insertScriptSchema>;
export type InsertReflection = z.infer<typeof insertReflectionSchema>;
