// index.js — Spirit v4.x Entrypoint
// ------------------------------------------------
import "dotenv/config";
import express from "express";
import rateLimit from "express-rate-limit";
import cors from "cors";

// Routers
import liveRouter from "./routes/live.js";
import healthRouter from "./routes/health.js";
import chatRouter from "./routes/chat.js";

const app = express();

// ------------------------------------------------
// CORS — Render/Lovable Final Integration
// ------------------------------------------------
const allowedOrigins = [
  "http://localhost:3000",
  /\.lovable\.app$/,
  /\.lovableproject\.com$/,
  /\.onrender\.com$/,
];

app.use(
  cors({
    origin: function (origin, callback) {
      // Allow server-to-server or curl
      if (!origin) return callback(null, true);

      const isAllowed = allowedOrigins.some((rule) =>
        rule instanceof RegExp ? rule.test(origin) : rule === origin
      );

      if (isAllowed) return callback(null, true);

      console.warn("[CORS] Blocked origin:", origin);
      return callback(new Error("Not allowed by CORS"));
    },
    credentials: false,
  })
);

// ------------------------------------------------
// Rate Limiting — Safety Layer (1 min window)
// ------------------------------------------------
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 50,
  message: { error: "Too many requests. Slow down." },
});
app.use(limiter);

// ------------------------------------------------
// Middleware
// ------------------------------------------------
app.use(express.json({ limit: "2mb" }));

// ------------------------------------------------
// Routes
// ------------------------------------------------
app.use("/health", healthRouter);
app.use("/chat", chatRouter);
app.use("/live", liveRouter); // MUST match frontend API_URL

// Root Confirmation Route
app.get("/", (_req, res) => {
  res.json({
    ok: true,
    service: "Spirit v4.x",
    msg: "You have arrived. Breathe. We begin.",
    ts: new Date().toISOString(),
  });
});

// ------------------------------------------------
// Start Server
// ------------------------------------------------
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
  console.log(`🜂 Spirit v4.x listening on port ${PORT}`);
});

export default app;
