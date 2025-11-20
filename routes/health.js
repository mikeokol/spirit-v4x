// routes/health.js — Spirit v5.1 Health Check

import express from "express";

const router = express.Router();

router.get("/", (_req, res) => {
  res.status(200).json({
    ok: true,
    service: "Spirit v5.1",
    message: "Health stable",
    ts: new Date().toISOString()
  });
});

export default router;