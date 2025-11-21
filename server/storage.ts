import {
  type ChatMessage,
  type InsertChatMessage,
  type Workout,
  type Script,
  type Reflection,
} from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  // Chat methods
  getChatMessages(): Promise<ChatMessage[]>;
  addChatMessage(message: InsertChatMessage): Promise<ChatMessage>;
  
  // Workout methods
  getWorkouts(): Promise<Workout[]>;
  addWorkout(workout: Workout): Promise<Workout>;
  
  // Script methods
  getScripts(): Promise<Script[]>;
  addScript(script: Script): Promise<Script>;
  
  // Reflection methods
  getReflections(): Promise<Reflection[]>;
  addReflection(reflection: Reflection): Promise<Reflection>;
}

export class MemStorage implements IStorage {
  private chatMessages: Map<string, ChatMessage>;
  private workouts: Map<string, Workout>;
  private scripts: Map<string, Script>;
  private reflections: Map<string, Reflection>;

  constructor() {
    this.chatMessages = new Map();
    this.workouts = new Map();
    this.scripts = new Map();
    this.reflections = new Map();
  }

  // Chat methods
  async getChatMessages(): Promise<ChatMessage[]> {
    return Array.from(this.chatMessages.values()).sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
  }

  async addChatMessage(insertMessage: InsertChatMessage): Promise<ChatMessage> {
    const id = randomUUID();
    const message: ChatMessage = {
      ...insertMessage,
      id,
      timestamp: new Date().toISOString(),
    };
    this.chatMessages.set(id, message);
    return message;
  }

  // Workout methods
  async getWorkouts(): Promise<Workout[]> {
    return Array.from(this.workouts.values()).sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }

  async addWorkout(workout: Workout): Promise<Workout> {
    this.workouts.set(workout.id, workout);
    return workout;
  }

  // Script methods
  async getScripts(): Promise<Script[]> {
    return Array.from(this.scripts.values()).sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }

  async addScript(script: Script): Promise<Script> {
    this.scripts.set(script.id, script);
    return script;
  }

  // Reflection methods
  async getReflections(): Promise<Reflection[]> {
    return Array.from(this.reflections.values()).sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }

  async addReflection(reflection: Reflection): Promise<Reflection> {
    this.reflections.set(reflection.id, reflection);
    return reflection;
  }
}

export const storage = new MemStorage();
