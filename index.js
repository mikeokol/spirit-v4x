// index.js — Spirit v7 Backend

import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import "dotenv/config";

// NEW — Spirit v7 Unified Cognitive Engine Route
import spiritRouter from "./routes/spirit.js";

const app = express();

// ===========================================
// GLOBAL MIDDLEWARE
// ===========================================

app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  })
);

app.use(
  bodyParser.json({
    limit: "5mb",
    strict: true,
  })
);

// ===========================================
// ROUTES
// ===========================================

// Spirit v7 Cognitive Engine
app.use("/spirit", spiritRouter);

// Root
app.get("/", (_req, res) => {
  res.json({
    ok: true,
    engine: "Spirit v7 Cognitive Engine",
    status: "online",
    endpoint: "/spirit",
  });
});

// 404 fallback
app.use((req, res) => {
  res.status(404).json({ ok: false, error: "Route not found", path: req.originalUrl });
});

// ===========================================
// SERVER START
// ===========================================

const PORT = process.env.PORT || 10000;

app.listen(PORT, () => {
  console.log(`✨ Spirit v7 Cognitive Engine running on port ${PORT}`);
});
