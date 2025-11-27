export interface ReflectionTemplate {
  id: string;
  name: string;
  category: string;
  preview: string;
}

export const REFLECTION_TEMPLATES: ReflectionTemplate[] = [
  {
    id: "default",
    name: "Default Reflection",
    category: "General",
    preview: `• What went well today?
• What challenged you?
• What did you learn about yourself?`,
  },

  {
    id: "morning_alignment",
    name: "Morning Alignment Scan",
    category: "Morning",
    preview: `• Who do I choose to be today?
• What identity will I embody?
• What is the one action that moves me forward?
• What energy do I want to carry into the day?`,
  },

  {
    id: "evening_growth_cycle",
    name: "Evening Growth Cycle",
    category: "Evening",
    preview: `• What identity did I embody today?
• What friction or resistance did I overcome?
• What pattern tried to pull me backward?
• How did I grow — even 1% — today?`,
  },

  {
    id: "identity_shift_journal",
    name: "Identity Shift Journal",
    category: "Identity",
    preview: `• What version of me showed up today?
• Where did I operate from habit vs intention?
• What identity trait strengthened today?
• What identity trait needs reinforcement tomorrow?`,
  },

  {
    id: "discipline_tracker",
    name: "Discipline Tracker",
    category: "Discipline",
    preview: `• What commitments did I keep today?
• What commitments did I break?
• Why did I break them? (No shame — just data)
• What one system change makes tomorrow easier?`,
  },

  {
    id: "creator_flow_scan",
    name: "Creator Flow Scan",
    category: "Creator",
    preview: `• What ideas or insights surfaced today?
• Where did flow come naturally?
• What blocked creation?
• What can I create tomorrow without friction?`,
  },

  {
    id: "mental_clarity",
    name: "Mental Clarity Reset",
    category: "Mind",
    preview: `• What is occupying my mind right now?
• What can I release?
• What is actually important today?
• What one thought brings me back to center?`,
  },

  {
    id: "emotional_debrief",
    name: "Emotional Debrief",
    category: "Emotions",
    preview: `• What emotion defined my day?
• Where did it come from?
• What did this emotion try to tell me?
• How can I respond instead of react tomorrow?`,
  },

  {
    id: "gratitude_expansion",
    name: "Gratitude Expansion",
    category: "Gratitude",
    preview: `• What am I grateful for today?
• Who supported me?
• What small moment brought me joy?
• What did I take for granted that deserves recognition?`,
  },

  {
    id: "performance_review",
    name: "High-Performance Review",
    category: "Performance",
    preview: `• What was my biggest win today?
• What slowed me down?
• What system failed?
• What system improved?
• What’s the one lever that moves everything forward?`,
  }
];