// routes/live.js — Spirit v5.1 Showcase Launch
// Live hybrid coaching (mind + body + identity)

import express from "express";
import { liveController } from "../controllers/liveController.js";

const router = express.Router();

// Single endpoint for live coaching sessions
router.post("/", liveController);

export default router;
