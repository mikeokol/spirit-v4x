# Spirit Backend v0.7  –  Continuity Ledger with Strategy Engine + LangSmith Trace

## What’s new
- **LangSmith trace decorator** on `run_daily_objective` – full observability of every generation run  
- **Strategy-aware daily objectives** – each prompt includes active strategy card + required signal keys  
- **UUID keys** – aligned with Supabase Auth  
- **Calibration Mode** – 3–5 question profile before any plan is generated  
- **Weekly Review** – `/api/strategic/review` pivots strategy based on 7-day signals  
- **Goal Switch Guardrail** – activating a new goal resets strategic mode  

## Stack
Python 3.12 + FastAPI + SQLModel + Postgres + LangGraph + OpenAI Structured Outputs + LangSmith  

## Quick start
```bash
cp .env.example .env              # fill secrets
pip install -r requirements.txt
alembic upgrade head
uvicorn spirit.main:app --reload
