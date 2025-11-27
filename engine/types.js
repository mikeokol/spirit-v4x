export type TaskType =
  | "fitness_plan"
  | "creator_content"
  | "hybrid_task"
  | "reflection_task"
  | "live_session"
  | "simple_chat"
  | "complex_chat";

export interface PlannerPlan {
  steps: string[];
}