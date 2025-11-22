// routes/hybrid.js — Spirit v5.1 Showcase Launch
// Combines Fitness + Creator into one unified endpoint

import express from "express";
import { hybridController } from "../controllers/hybridController.js";

const router = express.Router();

router.post("/", hybridController);

export default router;
