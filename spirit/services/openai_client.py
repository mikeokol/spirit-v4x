import os
from openai import AsyncOpenAI
from spirit.schemas.daily_objective import DailyObjectiveSchema

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def plan_daily_objective(prompt: str) -> dict:
    """Call OpenAI with Structured Output (response_format)."""
    completion = await client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are Spirit, a continuity ledger. Produce exactly one concrete daily objective."},
            {"role": "user", "content": prompt},
        ],
        response_format=DailyObjectiveSchema,
    )
    return completion.choices[0].message.parsed.model_dump()
