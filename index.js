// index.js — Spirit v5.1 Server Entrypoint

import express from "express";
import bodyParser from "body-parser";
import "dotenv/config";

import healthRouter from "./routes/health.js";
import chatRouter from "./routes/chat.js";

// NEW v5.1 routers
import fitnessRouter from "./routes/fitness.js";
import creatorRouter from "./routes/creator.js";
import reflectionRouter from "./routes/reflection.js";

const app = express();
app.use(bodyParser.json());

// Routes
app.use("/health", healthRouter);
app.use("/chat", chatRouter);
app.use("/fitness", fitnessRouter);
app.use("/creator", creatorRouter);
app.use("/reflection", reflectionRouter);

app.get("/", (_req, res) => {
  res.json({
    ok: true,
    service: "Spirit v5.1",
    message: "You have arrived. Breathe. We begin.",
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Spirit v5.1 backend running on port ${PORT}`));