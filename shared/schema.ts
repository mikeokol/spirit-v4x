import { z } from "zod";

// -----------------------------
// WORKOUTS
// -----------------------------
export const ExerciseSchema = z.object({
  name: z.string(),
  sets: z.number().nullable().optional(),
  reps: z.string().nullable().optional(),
  weight: z.string().nullable().optional(),
  notes: z.string().nullable().optional(),
});

export const WorkoutSchema = z.object({
  id: z.string().uuid(),
  goal: z.string(),
  fitness_level: z.string(),
  duration: z.string().nullable().optional(),
  created_at: z.string(),
  exercises: z.array(ExerciseSchema),
});

export const WorkoutInputSchema = z.object({
  goal: z.string(),
  fitness_level: z.string(),
  duration: z.string().optional(),
});

// -----------------------------
// REFLECTIONS
// -----------------------------
export const ReflectionSchema = z.object({
  id: z.string().uuid(),
  wins: z.array(z.string()),
  challenges: z.array(z.string()),
  growth: z.array(z.string()),
  tags: z.array(z.string()),
  created_at: z.string(),
  templateId: z.string().nullable().optional(),
});

export const ReflectionInputSchema = z.object({
  wins: z.array(z.string()),
  challenges: z.array(z.string()),
  growth: z.array(z.string()),
  tags: z.array(z.string()),
  templateId: z.string().optional(),
});

// -----------------------------
// SCRIPTS
// -----------------------------
export const ScriptSchema = z.object({
  id: z.string().uuid(),
  topic: z.string(),
  content: z.string(),
  created_at: z.string(),
});

export const ScriptInputSchema = z.object({
  topic: z.string(),
});

// -----------------------------
// HABIT ENGINE
// -----------------------------
export const HabitFrequencyEnum = z.enum(["daily", "weekly"]);

export const HabitSchema = z.object({
  id: z.string().uuid(),
  title: z.string(),
  category: z.string(),
  frequency: HabitFrequencyEnum,
  reason: z.string().nullable().optional(),
  start_date: z.string(),
  end_date: z.string().nullable().optional(),
  created_at: z.string(),
  updated_at: z.string().optional(),

  currentStreak: z.number().optional(),
  lastCompleted: z.string().nullable().optional(),
  completionRate7d: z.number().optional(),
});

export const HabitInputSchema = z.object({
  title: z.string().min(1),
  category: z.string().min(1),
  frequency: HabitFrequencyEnum,
  reason: z.string().optional(),
  start_date: z.string().optional(),
});

export const HabitLogSchema = z.object({
  id: z.string().uuid(),
  habit_id: z.string().uuid(),
  date: z.string(),
  completed: z.boolean(),
  created_at: z.string(),
});

// -----------------------------
// ANALYTICS
// -----------------------------
export const HabitStreakLeaderSchema = z.object({
  id: z.string().uuid(),
  title: z.string(),
  category: z.string(),
  currentStreak: z.number(),
});

export const AnalyticsSummarySchema = z.object({
  totalWorkouts: z.number(),
  totalReflections: z.number(),
  totalScripts: z.number(),
  totalHabits: z.number(),
  activeHabits: z.number(),

  workoutsThisWeek: z.number(),
  reflectionsThisWeek: z.number(),
  habitCompletions7d: z.number(),
  habitCompletionRate7d: z.number(),

  topIdentityTags: z.array(
    z.object({
      tag: z.string(),
      count: z.number(),
    })
  ),

  habitStreakLeaders: z.array(HabitStreakLeaderSchema),
});

// -----------------------------
// IDENTITY TAG DEEP DIVE
// -----------------------------
export const IdentityTagResponseSchema = z.object({
  tag: z.string(),

  reflections: z.array(
    z.object({
      id: z.string().uuid(),
      created_at: z.string(),
      wins: z.array(z.string()),
      challenges: z.array(z.string()),
      growth: z.array(z.string()),
      tags: z.array(z.string()),
    })
  ),

  habits: z.array(
    z.object({
      id: z.string().uuid(),
      title: z.string(),
      category: z.string(),
      frequency: HabitFrequencyEnum,
      reason: z.string().nullable(),
      start_date: z.string(),
    })
  ),

  count: z.number(),
  trendLast30: z.number(),

  insight: z.string().nullable().optional(),
});

export type IdentityTagDeepDive = z.infer<typeof IdentityTagResponseSchema>;

// -----------------------------
// WEEK VS WEEK ANALYTICS COMPARISON
// -----------------------------
export const WeekComparisonSchema = z.object({
  currentWeek: AnalyticsSummarySchema,
  previousWeek: AnalyticsSummarySchema,
  deltas: z.object({
    workouts: z.number(),
    reflections: z.number(),
    habitCompletionRate: z.number(), // 0–1 delta
    topIdentityTags: z.array(
      z.object({
        tag: z.string(),
        change: z.number(), // positive or negative
      })
    ),
  }),
});