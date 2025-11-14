// index.js — Spirit v4.x Core Server
// ----------------------------------
import "dotenv/config";
import express from "express";
import cors from "cors";

import healthRouter from "./routes/health.js";
import chatRouter from "./routes/chat.js";

const app = express();

// ─────────────────────────────────────────────
//  CORS — Lovable previews + localhost
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
      // Allow non-browser tools (like curl, server-to-server)
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
//  Middleware + Routes
// ─────────────────────────────────────────────
app.use(express.json());

app.use("/health", healthRouter);
app.use("/chat", chatRouter);

// Root route — simple sanity check
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
