// routes/reflection.js — Reflection API Endpoints

import express from "express";
import {
  generateReflection,
  getReflectionHistory
} from "../controllers/reflectionController.js";

const router = express.Router();

router.post("/", generateReflection);
router.get("/history", getReflectionHistory);

export default router;