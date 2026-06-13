"""
Entry point.

Usage:
    python main.py

The bot starts polling Telegram, the scheduler fires the pipeline
at times defined in PIPELINE_HOURS (config.py).
Use /run in the bot chat to trigger manually.

First-time Telethon setup:
    Fill in TELEGRAM_API_ID / TELEGRAM_API_HASH / TELEGRAM_PHONE in .env,
    then run ONCE before starting the bot:
        python auth_userbot.py
    This creates userbot.session in the project root.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

from aiogram.types import BotCommand

from database.models import init_db
from bot.telegram_bot import create_bot
from scheduler import setup_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy third-party loggers
for _noisy in ("httpx", "httpcore", "crewai", "litellm", "openai"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _cleanup_old_media(media_dir: str, max_age_days: int = 3) -> None:
    """
    Delete media files older than `max_age_days` from the media directory.
    Runs at every bot startup as a safety net for files that weren't cleaned up
    after publish/reject (e.g. after a crash or a manual restart).
    """
    media_path = Path(media_dir)
    if not media_path.exists():
        return
    cutoff = time.time() - max_age_days * 86400
    deleted = 0
    for f in media_path.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                deleted += 1
            except OSError:
                pass
    if deleted:
        logger.info("Startup cleanup: deleted %d media files older than %d days", deleted, max_age_days)


async def main() -> None:
    logger.info("Initialising database…")
    init_db()

    from config import MEDIA_DIR  # noqa: PLC0415
    _cleanup_old_media(MEDIA_DIR)

    # Validate critical configuration before starting
    from config import (
        ADMIN_CHAT_ID, LLM_PROVIDER, OPENROUTER_API_KEY,
        OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, TARGET_CHANNEL_ID,
        TELEGRAM_BOT_TOKEN,
    )
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if ADMIN_CHAT_ID == 0:
        missing.append("ADMIN_CHAT_ID (=0, must be set to your Telegram user_id)")
    if not TARGET_CHANNEL_ID:
        missing.append("TARGET_CHANNEL_ID")
    has_llm = bool(OPENROUTER_API_KEY or OPENAI_API_KEY or ANTHROPIC_API_KEY or DEEPSEEK_API_KEY)
    if not has_llm:
        missing.append("At least one LLM API key (OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)")
    if missing:
        print("FATAL: Missing required config values:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("Fix your .env file (copy from .env.example), then restart.", file=sys.stderr)
        sys.exit(1)

    logger.info("Creating bot…")
    bot, dp = create_bot()

    logger.info("Registering bot command menu…")
    await bot.set_my_commands([
        BotCommand(command="start",  description="Главное меню"),
        BotCommand(command="run",    description="Запустить дайджест вручную"),
        BotCommand(command="post",   description="✍️ Написать пост по своей новости"),
        BotCommand(command="status", description="Последние посты и дайджесты"),
        BotCommand(command="cancel", description="Отменить текущее действие"),
    ])

    logger.info("Setting up scheduler…")
    scheduler = setup_scheduler(bot)
    scheduler.start()

    logger.info("Bot started. Polling…")
    try:
        await dp.start_polling(bot, allowed_updates=["message", "callback_query"])
    finally:
        scheduler.shutdown(wait=False)
        await bot.session.close()

        # Gracefully disconnect the Telethon userbot client (if it was used this run)
        try:
            from parsers.telegram_userbot import close_userbot  # noqa: PLC0415
            await close_userbot()
        except ImportError:
            pass   # telethon not installed — nothing to close

        logger.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
