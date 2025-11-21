import {
  type ChatMessage,
  type InsertChatMessage,
  type Workout,
  type Script,
  type Reflection,
  chatMessages,
  workouts,
  scripts,
  reflections,
  insertWorkoutDBSchema,
  insertScriptDBSchema,
  insertReflectionDBSchema,
} from "@shared/schema";
import { db } from "./db";
import { desc, asc } from "drizzle-orm";

export interface IStorage {
  // Chat methods
  getChatMessages(): Promise<ChatMessage[]>;
  addChatMessage(message: InsertChatMessage): Promise<ChatMessage>;
  
  // Workout methods
  getWorkouts(): Promise<Workout[]>;
  addWorkout(title: string, exercises: any[]): Promise<Workout>;
  
  // Script methods
  getScripts(): Promise<Script[]>;
  addScript(title: string, type: string, content: string): Promise<Script>;
  
  // Reflection methods
  getReflections(): Promise<Reflection[]>;
  addReflection(title: string, wins: string[], challenges: string[], growth: string[], tags: string[]): Promise<Reflection>;
}

export class DatabaseStorage implements IStorage {
  // Chat methods
  async getChatMessages(): Promise<ChatMessage[]> {
    const messages = await db.select().from(chatMessages).orderBy(asc(chatMessages.timestamp));
    return messages.map(msg => ({
      id: msg.id,
      role: msg.role,
      content: msg.content,
      timestamp: msg.timestamp.toISOString(),
    }));
  }

  async addChatMessage(insertMessage: InsertChatMessage): Promise<ChatMessage> {
    const [message] = await db.insert(chatMessages).values({
      role: insertMessage.role,
      content: insertMessage.content,
    }).returning();
    
    return {
      id: message.id,
      role: message.role,
      content: message.content,
      timestamp: message.timestamp.toISOString(),
    };
  }

  // Workout methods
  async getWorkouts(): Promise<Workout[]> {
    const results = await db.select().from(workouts).orderBy(desc(workouts.timestamp));
    return results.map(workout => {
      // Parse and validate exercises from JSONB
      const exercises = insertWorkoutDBSchema.parse({
        title: workout.title,
        exercises: workout.exercises,
      }).exercises;
      
      return {
        id: workout.id,
        title: workout.title,
        exercises,
        timestamp: workout.timestamp.toISOString(),
      };
    });
  }

  async addWorkout(title: string, exercises: any[]): Promise<Workout> {
    // Validate exercises structure
    const validated = insertWorkoutDBSchema.parse({ title, exercises });
    
    const [result] = await db.insert(workouts).values({
      title: validated.title,
      exercises: validated.exercises as any,
    }).returning();
    
    return {
      id: result.id,
      title: result.title,
      exercises: result.exercises as any[],
      timestamp: result.timestamp.toISOString(),
    };
  }

  // Script methods
  async getScripts(): Promise<Script[]> {
    const results = await db.select().from(scripts).orderBy(desc(scripts.timestamp));
    return results.map(script => ({
      id: script.id,
      title: script.title,
      type: script.type,
      content: script.content,
      timestamp: script.timestamp.toISOString(),
    }));
  }

  async addScript(title: string, type: string, content: string): Promise<Script> {
    // Validate script structure
    const validated = insertScriptDBSchema.parse({ title, type, content });
    
    const [result] = await db.insert(scripts).values({
      title: validated.title,
      type: validated.type,
      content: validated.content,
    }).returning();
    
    return {
      id: result.id,
      title: result.title,
      type: result.type,
      content: result.content,
      timestamp: result.timestamp.toISOString(),
    };
  }

  // Reflection methods
  async getReflections(): Promise<Reflection[]> {
    const results = await db.select().from(reflections).orderBy(desc(reflections.timestamp));
    return results.map(reflection => {
      // Parse and validate reflection data from JSONB
      const validated = insertReflectionDBSchema.parse({
        title: reflection.title,
        wins: reflection.wins,
        challenges: reflection.challenges,
        growth: reflection.growth,
        tags: reflection.tags,
      });
      
      return {
        id: reflection.id,
        title: reflection.title,
        wins: validated.wins,
        challenges: validated.challenges,
        growth: validated.growth,
        tags: validated.tags,
        timestamp: reflection.timestamp.toISOString(),
      };
    });
  }

  async addReflection(title: string, wins: string[], challenges: string[], growth: string[], tags: string[]): Promise<Reflection> {
    // Validate reflection structure
    const validated = insertReflectionDBSchema.parse({ title, wins, challenges, growth, tags });
    
    const [result] = await db.insert(reflections).values({
      title: validated.title,
      wins: validated.wins as any,
      challenges: validated.challenges as any,
      growth: validated.growth as any,
      tags: validated.tags as any,
    }).returning();
    
    return {
      id: result.id,
      title: result.title,
      wins: result.wins as string[],
      challenges: result.challenges as string[],
      growth: result.growth as string[],
      tags: result.tags as string[],
      timestamp: result.timestamp.toISOString(),
    };
  }
}

export const storage = new DatabaseStorage();
