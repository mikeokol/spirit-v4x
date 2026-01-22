import os
from openai import AsyncOpenAI
from spirit.config import settings
from spirit.schemas.daily_objective import DailyObjectiveSchema

client = AsyncOpenAI(api_key=settings.openai_api_key)

async def plan_daily_objective(prompt: str) -> dict:
    """Call OpenAI Responses API with forced schema."""
    completion = await client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are Spirit, a continuity ledger. Return only JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format=DailyObjectiveSchema,
    )
    return completion.choices[0].message.parsed.model_dump()
