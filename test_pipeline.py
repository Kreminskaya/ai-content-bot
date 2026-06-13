"""One-shot pipeline test — run and watch the output."""
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
)

from scheduler import run_pipeline_job
from bot.telegram_bot import create_bot


async def main():
    bot, _dp = create_bot()
    print(">>> Starting pipeline job…")
    try:
        await run_pipeline_job(bot)
        print(">>> Done.")
    finally:
        await bot.session.close()


asyncio.run(main())
