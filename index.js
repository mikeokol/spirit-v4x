// index.js — Spirit v4.x Entrypoint
// ---------------------------------
import "dotenv/config";
import express from "express";
import rateLimit from "express-rate-limit";
import cors from "cors";

import healthRouter from "./routes/health.js";
import chatRouter from "./routes/chat.js";

const app = express();

// ---------------------------------------------
// CORS — Finalized Patch (Lovable-proof)
// ---------------------------------------------
const allowedOrigins = [
  "http://localhost:3000",
  /\.lovable\.app$/,
  /\.lovableproject\.com$/,
];

app.use(
  cors({
    origin: function (origin, callback) {
      if (!origin) return callback(null, true);

      const allowed = allowedOrigins.some((rule) =>
        rule instanceof RegExp ? rule.test(origin) : rule === origin
      );

      if (allowed) return callback(null, true);

      console.log("[CORS] Blocked origin:", origin);
      return callback(new Error("Not allowed by CORS"));
    },
    credentials: false,
  })
);

// ---------------------------------------------
// Rate Limiting — Protect API From Abuse
// ---------------------------------------------
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 50,
});
app.use(limiter);

// ---------------------------------------------
// Middleware
// ---------------------------------------------
app.use(express.json());

// ---------------------------------------------
// Routes
// ---------------------------------------------
app.use("/health", healthRouter);
app.use("/chat", chatRouter);

// Root confirmation route
app.get("/", (_req, res) => {
  res.json({
    ok: true,
    service: "Spirit v4.x",
    msg: "You have arrived. Breathe. We begin.",
    ts: new Date().toISOString(),
  });
});

// ---------------------------------------------
// Start Server
// ---------------------------------------------
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
  console.log(`🜂 Spirit v4.x listening on port ${PORT}`);
});

export default app;
    