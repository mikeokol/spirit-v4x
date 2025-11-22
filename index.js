// index.js — Spirit v5.1 Showcase Launch Backend

import express from "express";
import bodyParser from "body-parser";
import "dotenv/config";

// Core routes
import chatRouter from "./routes/chat.js";
import fitnessRouter from "./routes/fitness.js";
import creatorRouter from "./routes/creator.js";
import hybridRouter from "./routes/hybrid.js";
import liveRouter from "./routes/live.js";
import healthRouter from "./routes/health.js";

const app = express();
app.use(bodyParser.json());

// Routes
app.use("/health", healthRouter);
app.use("/chat", chatRouter);
app.use("/fitness", fitnessRouter);
app.use("/creator", creatorRouter);
app.use("/hybrid", hybridRouter);
app.use("/live-session", liveRouter);

// Root route (sanity check)
app.get("/", (_req, res) => {
  res.json({
    ok: true,
    service: "Spirit v5.1 Showcase",
    message: "You have arrived. Breathe. We begin."
  });
});

// Server startup
const PORT = process.env.PORT || 3000;
app.listen(PORT, () =>
  console.log(`✨ Spirit v5.1 Showcase backend running on port ${PORT}`)
);
