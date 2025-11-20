// fitness.js — v5.1 Fitness API Routes

import express from "express";
import {
  setFitnessProfile,
  getFitnessBlock,
  generateFitnessBlock,
  getSessionForDay,
  logSession
} from "../controllers/fitnessController.js";

const router = express.Router();

router.post("/profile", setFitnessProfile);
router.get("/block", getFitnessBlock);
router.post("/block/regenerate", generateFitnessBlock);
router.get("/session/:day", getSessionForDay);
router.post("/session/log", logSession);

export default router;