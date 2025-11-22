// routes/fitness.js — Spirit v5.1 Showcase Launch
// Single-endpoint Fitness Engine for rapid testing & clean UX

import express from "express";
import { fitnessController } from "../controllers/fitnessController.js";

const router = express.Router();

// All fitness logic goes through one clean endpoint
router.post("/", fitnessController);

export default router;
