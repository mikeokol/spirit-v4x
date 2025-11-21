export type ReflectionTemplate = {
  id: string;
  name: string;
  description: string;
  prompt: string;
};

export const reflectionTemplates: ReflectionTemplate[] = [
  {
    id: "general",
    name: "General Reflection",
    description: "A flexible reflection for any time period",
    prompt: `Generate a guided reflection for {period}{focus}.

Return a JSON object with:
- title: A reflective title for this period
- wins: Array of 3-5 wins or achievements
- challenges: Array of 3-5 challenges or obstacles faced
- growth: Array of 3-5 growth areas or lessons learned
- tags: Array of 3-5 relevant theme tags

Make it thoughtful, specific, and encouraging.`
  },
  {
    id: "weekly",
    name: "Weekly Review",
    description: "Review your week's progress and plan ahead",
    prompt: `Generate a weekly reflection for {period}{focus}.

Return a JSON object with:
- title: A title for this week's reflection
- wins: Array of 3-5 specific accomplishments from this week (include what you learned or how it felt)
- challenges: Array of 3-5 obstacles you faced (include what made them difficult and how you handled them)
- growth: Array of 3-5 insights or lessons for next week (actionable takeaways)
- tags: Array of 3-5 themes that defined this week

Focus on: What went well? What was hard? What will I do differently next week?`
  },
  {
    id: "monthly",
    name: "Monthly Goals Review",
    description: "Assess monthly goals and set intentions",
    prompt: `Generate a monthly reflection for {period}{focus}.

Return a JSON object with:
- title: A title for this month's reflection
- wins: Array of 3-5 major wins or milestones this month (quantify progress where possible)
- challenges: Array of 3-5 setbacks or learning opportunities (what didn't go as planned?)
- growth: Array of 3-5 goals or focus areas for next month (specific, measurable intentions)
- tags: Array of 3-5 key themes or focus areas from this month

Consider: Did you meet your goals? What surprised you? What needs more attention?`
  },
  {
    id: "quarterly",
    name: "Quarterly Review",
    description: "Big picture reflection on the past quarter",
    prompt: `Generate a quarterly reflection for {period}{focus}.

Return a JSON object with:
- title: A title for this quarter's reflection
- wins: Array of 3-5 significant achievements or breakthroughs (how did these impact your bigger goals?)
- challenges: Array of 3-5 major obstacles or patterns you noticed (what repeated themes emerged?)
- growth: Array of 3-5 strategic priorities for next quarter (aligned with long-term vision)
- tags: Array of 3-5 overarching themes from this quarter

Think big: What progress did you make on long-term goals? What patterns emerged? What needs to shift?`
  },
  {
    id: "gratitude",
    name: "Gratitude Practice",
    description: "Focus on appreciation and positive moments",
    prompt: `Generate a gratitude-focused reflection for {period}{focus}.

Return a JSON object with:
- title: A gratitude-themed title
- wins: Array of 5-7 things you're grateful for from this period (big or small - people, moments, experiences)
- challenges: Array of 2-3 difficult moments that taught you something valuable (silver linings)
- growth: Array of 3-5 ways you can cultivate more gratitude or appreciation going forward
- tags: Array of 3-5 themes related to gratitude and appreciation

Emphasize: What brought you joy? Who supported you? What unexpected blessings appeared?`
  },
  {
    id: "creative",
    name: "Creative Reflection",
    description: "Reflect on creative work and projects",
    prompt: `Generate a creative reflection for {period}{focus}.

Return a JSON object with:
- title: A creative-focused title
- wins: Array of 3-5 creative breakthroughs, finished projects, or inspired moments
- challenges: Array of 3-5 creative blocks, struggles, or experiments that didn't work
- growth: Array of 3-5 ideas to explore or skills to develop in your creative practice
- tags: Array of 3-5 creative themes or mediums you engaged with

Consider: What inspired you? When did you feel most creative? What wants to be expressed next?`
  },
  {
    id: "wellness",
    name: "Wellness Check-In",
    description: "Reflect on physical and mental well-being",
    prompt: `Generate a wellness-focused reflection for {period}{focus}.

Return a JSON object with:
- title: A wellness-themed title
- wins: Array of 3-5 positive health habits, self-care moments, or wellness achievements
- challenges: Array of 3-5 wellness struggles or areas that need attention (physical, mental, emotional)
- growth: Array of 3-5 wellness goals or practices to prioritize moving forward
- tags: Array of 3-5 wellness themes (sleep, movement, nutrition, stress, relationships, etc.)

Reflect on: How did you care for yourself? What drained your energy? What would support your well-being?`
  },
  {
    id: "relationships",
    name: "Relationships Reflection",
    description: "Focus on connections and social well-being",
    prompt: `Generate a relationships-focused reflection for {period}{focus}.

Return a JSON object with:
- title: A relationships-themed title
- wins: Array of 3-5 meaningful connections, conversations, or relationship growth moments
- challenges: Array of 3-5 relationship struggles, conflicts, or areas needing attention
- growth: Array of 3-5 ways to nurture relationships or set healthier boundaries
- tags: Array of 3-5 relationship themes (family, friends, work, community, boundaries, etc.)

Consider: Who energized you? What relationships need care? How can you show up better for others?`
  },
  {
    id: "learning",
    name: "Learning & Development",
    description: "Track skills, knowledge, and growth",
    prompt: `Generate a learning-focused reflection for {period}{focus}.

Return a JSON object with:
- title: A learning-themed title
- wins: Array of 3-5 new skills learned, knowledge gained, or concepts mastered
- challenges: Array of 3-5 learning struggles, confusing topics, or skills that need more practice
- growth: Array of 3-5 learning goals or areas to explore next (courses, books, projects, mentors)
- tags: Array of 3-5 learning topics or skill areas

Reflect on: What did you learn? What sparked curiosity? What do you want to understand better?`
  }
];

export function getTemplateById(id: string): ReflectionTemplate | undefined {
  return reflectionTemplates.find(t => t.id === id);
}

export function formatTemplatePrompt(template: ReflectionTemplate, period: string, focus?: string): string {
  const focusText = focus ? ` focusing on ${focus}` : "";
  return template.prompt
    .replace("{period}", period)
    .replace("{focus}", focusText);
}
