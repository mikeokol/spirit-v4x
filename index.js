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
  /\.lovable\.app$/, // any Lovable preview subdomain
];

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin) return callback(null, true); // curl / server-to-server

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
//  Global middleware
// ─────────────────────────────────────────────
app.use(express.json());

// Simple rate limiter for /chat (per IP)
const chatLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 20,             // 20 requests / minute / IP
  standardHeaders: true,
  legacyHeaders: false,
});

// ─────────────────────────────────────────────
//  Routes
// ─────────────────────────────────────────────
app.use("/health", healthRouter);
app.use("/chat", chatLimiter, chatRouter);

// Root route — sanity check
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
