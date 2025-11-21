# Spirit v5.1 - AI Companion Application

## Overview

Spirit v5.1 is a full-stack AI-powered companion application that provides four core features: conversational chat, personalized workout generation, creator script generation, and guided reflections. Built as a modern web application with a React frontend and Express backend, it leverages OpenAI's GPT models (via Replit's AI Integrations service) to generate intelligent, context-aware content across all features.

The application follows a clean, productivity-focused design inspired by Linear and ChatGPT's UI patterns, emphasizing clarity and functional simplicity. It's designed to be a comprehensive personal assistant tool that helps users with conversation, fitness planning, content creation, and self-reflection.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework & Build Tools:**
- React 18 with TypeScript for type-safe component development
- Vite as the build tool and development server for fast HMR and optimized production builds
- Wouter for lightweight client-side routing (routes: `/`, `/workouts`, `/scripts`, `/reflections`)

**UI Component System:**
- shadcn/ui component library (New York style variant) built on Radix UI primitives
- Tailwind CSS for utility-first styling with extensive custom design system
- Custom CSS variables system for theme consistency (light/dark mode support)
- Design system based on Inter font family with JetBrains Mono for monospace content

**State Management:**
- TanStack Query (React Query) for server state management, caching, and data synchronization
- React Hook Form with Zod for form state and validation
- Local component state with React hooks for UI-specific state

**Application Structure:**
- Single-page application with sidebar navigation pattern
- Main layout uses SidebarProvider with collapsible navigation
- Four primary feature pages, each with tab-based interfaces (Generate/History)
- Responsive design with mobile-first approach

### Backend Architecture

**Server Framework:**
- Express.js with TypeScript for the HTTP server
- ESM module system throughout the codebase
- Custom middleware for request logging and JSON body parsing with raw body preservation

**API Design:**
- RESTful API endpoints organized by feature:
  - `/api/chat` - Chat message history and creation
  - `/api/workouts` - Workout generation and retrieval
  - `/api/scripts` - Script generation and retrieval
  - `/api/reflections` - Reflection generation and retrieval
- Each feature follows a consistent pattern: GET for list retrieval, POST for generation
- Schema validation using Zod schemas from shared types

**Data Storage:**
- In-memory storage implementation (`MemStorage` class) using Map data structures
- Designed with an `IStorage` interface for easy migration to persistent storage (Drizzle ORM configuration present for future PostgreSQL integration)
- Each data entity (ChatMessage, Workout, Script, Reflection) has its own Map store
- UUID-based entity identification

**AI Integration:**
- OpenAI SDK configured to use Replit's AI Integrations service
- GPT-5.1 model for all AI generation tasks
- Separate generation functions for each feature type:
  - `generateChatResponse()` - Conversational responses
  - `generateWorkout()` - Structured workout plans with exercises
  - `generateScript()` - Content scripts for various platforms
  - `generateReflection()` - Guided reflection prompts and frameworks
- System prompts customized per feature to ensure appropriate output formatting

**Development Tools:**
- Vite middleware integration for development mode HMR
- Custom logging system with timestamp formatting
- Replit-specific plugins for cartographer and dev banner (development only)

### Data Schema & Validation

**Shared Schema Design:**
- Centralized schema definitions in `shared/schema.ts` using Zod
- Separate schemas for complete entities vs. insert operations (omitting auto-generated fields)
- Type inference from Zod schemas ensures type safety across frontend and backend

**Entity Types:**
- ChatMessage: role (user/assistant), content, timestamp, id
- Workout: title, exercises array (name, sets, reps, duration, notes), timestamp, id
- Script: title, type, content, timestamp, id
- Reflection: title, framework, content, timestamp, id

**Validation Strategy:**
- Input validation schemas for user-submitted data
- Runtime validation at API boundaries
- TypeScript types derived from validated schemas

### Configuration & Build System

**TypeScript Configuration:**
- Strict mode enabled with comprehensive type checking
- Path aliases configured for clean imports (`@/`, `@shared/`, `@assets/`)
- ESNext module system with bundler resolution
- Separate include paths for client, server, and shared code

**Build Process:**
- Development: Vite dev server with Express API proxy
- Production: 
  - Frontend bundled via Vite to `dist/public`
  - Backend bundled via esbuild to `dist/index.js`
  - Static file serving from bundled frontend

**Environment Configuration:**
- Database configuration present (Drizzle) but not actively used
- AI Integrations credentials via environment variables
- Support for Replit-specific environment detection

### Design System

**Typography Scale:**
- H1: text-4xl (feature titles)
- H2: text-2xl (section headers)
- H3: text-lg (card titles)
- Body: text-base
- Small: text-sm
- Tiny: text-xs

**Spacing System:**
- Tailwind spacing units: 2, 4, 8, 12, 16
- Consistent padding/margin patterns across components
- Content max-width: 4xl for reading, 3xl for chat interface

**Color System:**
- HSL-based color variables with alpha channel support
- Separate color definitions for light and dark modes
- Semantic color naming (primary, secondary, muted, accent, destructive)
- Card and popover variants for layered UI components

**Component Patterns:**
- Outline buttons with subtle borders for secondary actions
- Card-based layouts with consistent border and shadow styling
- Toast notifications for user feedback
- Skeleton loading states for async operations

## External Dependencies

### Third-Party Services

**Replit AI Integrations:**
- Purpose: Provides OpenAI-compatible API access without requiring personal OpenAI API key
- Integration: OpenAI SDK configured with custom base URL and API key from environment
- Models: GPT-5.1 for all AI generation features
- Usage: Chat responses, workout generation, script creation, reflection frameworks

**Neon Database (Configured but not actively used):**
- Purpose: PostgreSQL database provider
- Integration: Drizzle ORM configuration and schema present in codebase
- Status: Currently using in-memory storage; database setup prepared for future migration
- Connection: Via `@neondatabase/serverless` driver with connection pooling

### Frontend Libraries

**UI & Styling:**
- @radix-ui/* (v1.x-2.x) - 20+ primitive components for accessible UI patterns
- Tailwind CSS (v3.x) - Utility-first CSS framework
- class-variance-authority - Type-safe variant styling
- lucide-react - Icon library

**Forms & Validation:**
- react-hook-form - Performant form state management
- @hookform/resolvers - Integration with Zod validation
- zod - Schema validation library used across frontend/backend

**State & Data:**
- @tanstack/react-query (v5.60+) - Server state management
- wouter - Lightweight routing (~1.2KB)

**UI Enhancement:**
- embla-carousel-react - Carousel/slider component
- cmdk - Command palette component
- date-fns - Date formatting and manipulation

### Backend Libraries

**Server & Database:**
- express - Web server framework
- drizzle-orm - Type-safe ORM (configured for future use)
- drizzle-zod - Zod schema generation from Drizzle schemas
- connect-pg-simple - PostgreSQL session store (configured)

**AI & Utilities:**
- openai - Official OpenAI SDK
- nanoid - Unique ID generation
- zod - Schema validation (shared with frontend)

### Development Tools

**Build & Development:**
- vite - Frontend build tool and dev server
- @vitejs/plugin-react - React support for Vite
- esbuild - Backend bundler
- tsx - TypeScript execution for development
- typescript - Type checking

**Replit-Specific:**
- @replit/vite-plugin-runtime-error-modal - Error overlay
- @replit/vite-plugin-cartographer - Development tooling
- @replit/vite-plugin-dev-banner - Development banner

**Code Quality:**
- drizzle-kit - Database migration tooling
- PostCSS with autoprefixer - CSS processing

### Package Management
- npm with package-lock.json for deterministic installs
- ESM module system throughout the project
- Shared dependencies between client and server via monorepo-style structure