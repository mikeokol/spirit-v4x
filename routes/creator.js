// creator.js — v5.1 Creator API Routes

import express from "express";
import {
  setCreatorProfile,
  generateCreatorScript,
  generateContentSprint
} from "../controllers/creatorController.js";

const router = express.Router();

router.post("/profile", setCreatorProfile);
router.post("/script", generateCreatorScript);
router.post("/sprint", generateContentSprint);

export default router;