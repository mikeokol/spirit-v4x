<<<<<<< HEAD
// index.js — Spirit v4.x Core Server (with rate limiting)
// -------------------------------------------------------
=======
// index.js — Spirit Core v4.x (Server Entrypoint)
// ------------------------------------------------------

import express from "express";
import cors from "cors";
>>>>>>> 6fae1f4 (Upgrade Spirit backend to v4.x entrypoint)
import "dotenv/config";
import express from "express";
import cors from "cors";
import rateLimit from "express-rate-limit";

import healthRouter from "./routes/health.js";
import chatRouter from "./routes/chat.js";

const app = express();

<<<<<<< HEAD
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
=======
// JSON body parser
app.use(express.json({ limit: "1mb" }));

// CORS — required for Lovable frontend
app.use(
  cors({
    origin: "*", // allow all frontend origins
>>>>>>> 6fae1f4 (Upgrade Spirit backend to v4.x entrypoint)
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);
<<<<<<< HEAD

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
=======
>>>>>>> 6fae1f4 (Upgrade Spirit backend to v4.x entrypoint)

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
<<<<<<< HEAD
    message: "You have arrived. Breathe. We begin.",
=======
    message: "You have arrived. Spirit is online.",
>>>>>>> 6fae1f4 (Upgrade Spirit backend to v4.x entrypoint)
    ts: new Date().toISOString(),
  });
});

// ─────────────────────────────────────────────
//  Start Server
// ─────────────────────────────────────────────
<<<<<<< HEAD
const PORT = process.env.PORT || 3000;
=======
const PORT = process.env.PORT || 10000;
app.listen(PORT, () =>
  console.log(`✅ Spirit API (v4.x) running on port ${PORT}`)
);
>>>>>>> 6fae1f4 (Upgrade Spirit backend to v4.x entrypoint)

app.listen(PORT, () => {
  console.log(`🜂 Spirit v4.x listening on port ${PORT}`);
});

export default app;
