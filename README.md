# Spirit Backend v0.6  –  Continuity Ledger with Strategy Engine

## What’s new
- **Strategy Engine** – 12 deterministic strategy cards (business, career, fitness, creator)  
- **Weekly Review** – `/api/strategic/review` pivots strategy based on 7-day signals  
- **Goal Switch Guardrail** – activating a new goal resets strategic mode  
- **Strategy-aware daily objectives** – each objective includes required signal keys to log  

## Stack
Python 3.12 + FastAPI + SQLModel + Postgres + LangGraph + OpenAI Structured Outputs  

## Quick start
```bash
cp .env.example .env              # fill secrets
pip install -r requirements.txt
alembic upgrade head
uvicorn spirit.main:app --reload
