// index.js — Spirit v5.0 Backend Entrypoint
// -----------------------------------------
import "dotenv/config";
import express from "express";
import cors from "cors";

import healthRouter from "./routes/health.js";
import chatRouter from "./routes/chat.js";
import liveRouter from "./routes/live.js";
import sessionRouter from "./routes/session.js";

const app = express();

// CORS for Lovable + local dev
app.use(
  cors({
    origin: [
      "http://localhost:3000",
      /\.lovable\.app$/,
      "https://spirit-ai-coach-creator.lovable.app",
    ],
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type", "Authorization", "x-user-id"],
  })
);

app.use(express.json());

// ROUTES
app.use("/health", healthRouter);
app.use("/chat", chatRouter);
app.use("/live", liveRouter);
app.use("/session", sessionRouter);

// ROOT: sanity check
app.get("/", (_req, res) => {
  res.json({
    ok: true,
    service: "Spirit v5.0",
    message: "You have arrived. Breathe. We begin.",
    ts: new Date().toISOString(),
  });
});

// START SERVER
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
  console.log(`🔥 Spirit v5.0 active on port ${PORT}`);
});
