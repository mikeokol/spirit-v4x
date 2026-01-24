import logging
from langsmith import traceable
from openai import AsyncOpenAI
from spirit.config import settings
from spirit.schemas.daily_objective import DailyObjectiveSchema

client = AsyncOpenAI(api_key=settings.openai_api_key)
logger = logging.getLogger("spirit")

@traceable(run_type="llm", name="plan_daily_objective")
async def plan_daily_objective(prompt: str) -> dict:
    completion = await client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are Spirit. Output must match the schema."},
            {"role": "user", "content": prompt},
        ],
        response_format=DailyObjectiveSchema,
    )
    parsed = completion.choices[0].message.parsed
    logger.info("structured_output_parsed", extra={"is_schema": isinstance(parsed, DailyObjectiveSchema)})
    return parsed.model_dump()
