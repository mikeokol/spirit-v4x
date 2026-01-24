# Spirit Backend v0.4  –  Continuity Ledger with Calibration Mode

## What’s new
- **UUID primary keys** – aligned with Supabase Auth  
- **Calibration Mode** – 3–5 question profile before any plan is generated  
- **Reality-Anchor policy** – driver math + bottleneck pick → daily objective  
- **LangSmith traces** – every node + LLM call observable  
- **Guardrails 1–22** – ownership, idempotency, date safety, fallback logic  

## Stack
Python 3.12 + FastAPI + SQLModel + Postgres + LangGraph + OpenAI Structured Outputs  

## Quick start
```bash
cp .env.example .env              # fill secrets
pip install -r requirements.txt
alembic upgrade head
uvicorn spirit.main:app --reload
