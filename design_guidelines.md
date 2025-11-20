# Spirit v5.1 Design Guidelines

## Design Approach

**System**: Hybrid approach drawing from Linear's clean productivity aesthetics and ChatGPT's conversational UI patterns. This combines modern minimalism with functional clarity optimized for AI-powered tools.

**Core Principles**:
- Clarity over decoration - every element serves the user's workflow
- Generous spacing for cognitive breathing room
- Consistent patterns across all four features (chat, workouts, scripts, reflections)
- Fast, responsive interactions with subtle feedback

---

## Typography

**Font Stack**: 
- Primary: Inter (Google Fonts) - UI elements, body text
- Monospace: JetBrains Mono - code blocks, technical content

**Hierarchy**:
- H1: text-4xl font-semibold (Feature titles)
- H2: text-2xl font-semibold (Section headers)
- H3: text-lg font-medium (Card titles, subsections)
- Body: text-base (Main content, chat messages)
- Small: text-sm (Metadata, timestamps, labels)
- Tiny: text-xs (Helper text, hints)

---

## Layout System

**Spacing Primitives**: Use Tailwind units of 2, 4, 8, 12, 16
- Micro spacing: p-2, gap-2
- Standard spacing: p-4, gap-4, m-4
- Section spacing: p-8, py-12, gap-8
- Page margins: px-4 on mobile, px-8 on desktop

**Container Strategy**:
- App shell: Full viewport with sidebar navigation
- Content areas: max-w-4xl for optimal reading/interaction
- Chat interface: max-w-3xl centered for conversational flow

---

## Core Components

### Navigation Structure
**Sidebar Navigation** (Left-aligned, persistent):
- Width: w-64 on desktop, collapsible to icon-only on mobile
- Navigation items with icon + label
- Clear active state indication
- Sections: Chat, Workouts, Scripts, Reflections

### Chat Interface (Primary Feature)
**Layout**:
- Message thread: Conversational layout with alternating alignment
- User messages: Right-aligned with distinctive treatment
- Spirit messages: Left-aligned, full-width responses
- Input area: Fixed bottom position, sticky behavior

**Message Components**:
- Avatar indicators for Spirit vs User
- Timestamp metadata (text-xs)
- Message bubbles with rounded-2xl borders
- Code blocks with syntax highlighting (use Prism.js)

**Input Area**:
- Multi-line textarea with auto-expand (max 5 lines before scroll)
- Send button (primary CTA)
- Character/token counter
- Attachment/options menu

### Generation Features (Workouts, Scripts, Reflections)

**Form Layout**:
- Single column on mobile, two-column grid on desktop (grid-cols-1 md:grid-cols-2)
- Input fields with clear labels above
- Helper text below inputs
- Generate button as primary CTA

**Results Display**:
- Card-based layout (rounded-xl, distinct border)
- Header with title and metadata
- Content area with proper typography hierarchy
- Action buttons: Copy, Share, Save
- Regenerate option clearly visible

**Workout Cards**:
- Exercise name (font-medium)
- Sets/reps/duration (grid layout)
- Notes section
- Progress indicators if applicable

**Script Cards**:
- Title and type indicator
- Formatted script content (monospace for code)
- Copy-to-clipboard functionality
- Syntax highlighting for code snippets

**Reflection Cards**:
- Date/time header
- Categorized sections (Wins, Challenges, Growth)
- Bullet points or short paragraphs
- Tags for themes

---

## Form Elements

**Text Inputs**:
- Height: h-12 for single-line, min-h-24 for textareas
- Padding: px-4 py-3
- Rounded: rounded-lg
- Focus states with ring treatment

**Buttons**:
- Primary: px-6 py-3, rounded-lg, font-medium
- Secondary: Similar sizing with distinct treatment
- Icon buttons: w-10 h-10, rounded-lg
- Disabled states clearly indicated

**Select/Dropdown**:
- Match text input styling
- Clear dropdown indicator
- Keyboard navigation support

---

## Interaction Patterns

**Loading States**:
- Skeleton screens for content areas
- Spinner for in-progress generations
- Pulse animation for loading elements

**Empty States**:
- Centered icon + message
- Clear call-to-action
- Brief instructions or benefits

**Error Handling**:
- Inline validation messages (text-sm)
- Toast notifications for system errors (top-right corner)
- Retry options for failed requests

---

## Icons

**Library**: Heroicons (via CDN)
- Navigation: outline style
- Actions: solid style for primary, outline for secondary  
- Status indicators: mini style

**Icon Sizing**:
- Navigation: w-6 h-6
- Buttons: w-5 h-5
- Inline: w-4 h-4

---

## Responsive Behavior

**Breakpoints**:
- Mobile: Single column, collapsed sidebar
- Tablet (md): Two-column grids where applicable
- Desktop (lg): Full sidebar, optimal spacing

**Mobile Optimizations**:
- Bottom navigation bar for main features
- Full-screen modals instead of popovers
- Touch-friendly targets (min 44px)
- Simplified layouts, prioritize vertical flow

---

## Images

**Profile/Avatar Images**:
- Spirit avatar: 40x40px circular icon (abstract AI representation)
- User avatar: 32x32px circular
- Placeholder for user: initials in circle

**Feature Illustrations** (Optional):
- Empty state illustrations for each feature (simple, abstract)
- Size: max-w-xs centered above empty state text

**No Hero Image**: This is a utility application - users navigate directly to functional areas. No marketing hero needed.

---

## Animation Guidelines

**Minimal, Purposeful Motion**:
- Page transitions: Simple fade (150ms)
- Message appearance: Slide-up with fade (200ms)
- Button feedback: Scale down slightly on click
- NO auto-playing animations
- NO scroll-triggered effects