// index.js — Spirit v4.x Core Server (with rate limiting)
// -------------------------------------------------------
import "dotenv/config";
import express from "express";
import cors from "cors";
import rateLimit from "express-rate-limit";

import healthRouter from "./routes/health.js";
import chatRouter from "./routes/chat.js";

const app = express();

// ─────────────────────────────────────────────
//  CORS — only your apps + localhost
// ─────────────────────────────────────────────
const allowedOrigins = [
  "http://localhost:3000",
  "http://localhost:4173",
  "https://spirit-ai-coach-creator.lovable.app",
  "https://spirit-whisperer-ui.lovable.app",
  "https://spirit-symbiosis-web.lovable.app",
  /\.lovable\.app$/,
];

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin) return callback(null, true);

      const allowed = allowedOrigins.some((o) =>
        o instanceof RegExp ? o.test(origin) : o === origin
      );

      if (allowed) return callback(null, true);

      console.warn("[CORS] Blocked origin:", origin);
      return callback(new Error("Not allowed by CORS"));
    },
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

// ─────────────────────────────────────────────
//  JSON Parsing
// ─────────────────────────────────────────────
app.use(express.json());

// ─────────────────────────────────────────────
//  Rate Limiting — Protect /chat
// ─────────────────────────────────────────────
const chatLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 20,
  standardHeaders: true,
  legacyHeaders: false,
});

// ─────────────────────────────────────────────
//  Routes
// ─────────────────────────────────────────────
app.use("/health", healthRouter);
app.use("/chat", chatLimiter, chatRouter);

// ─────────────────────────────────────────────
//  Root Route (Sanity)
// ─────────────────────────────────────────────
app.get("/", (_req, res) => {
  res.status(200).json({
    ok: true,
    service: "Spirit v4.x",
    message: "You have arrived. Breathe. We begin.",
    ts: new Date().toISOString(),
  });
});

// ─────────────────────────────────────────────
//  Start Server
// ─────────────────────────────────────────────
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`🜂 Spirit v4.x listening on port ${PORT}`);
});

export default app;