// routes/creator.js — Spirit v5.1 Showcase Launch
// Single-endpoint Creator Engine for rapid testing

import express from "express";
import { creatorController } from "../controllers/creatorController.js";

const router = express.Router();

router.post("/", creatorController);

export default router;
