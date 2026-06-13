"""
APScheduler: runs the "Воронка" pipeline once or twice a day.

Pipeline steps (new architecture):
  1. Fetch posts from ALL sources:
       • Telegram channels (Telethon primary, HTTP fallback)
       • RSS feeds (feedparser)
       • Reddit /new.json
  2. Apply hard filter (filter_ads) — strip promo / low-quality posts
  3. Run DigestCrew (Analyst agent):
       → JSON digest with Top-5 Hard News + Top-5 Useful topics
  4. Save digest to SQLite with status = 'pending'
  5. Send "📰 Дайджест за [Дата]" to admin with 10 topic buttons

When admin clicks a topic button (handled in bot/handlers.py):
  6. WriterCrew generates 3 post variants for the chosen topic
  7. Admin picks a variant → published to channel
"""

import asyncio
import logging

from aiogram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from agents.digest_crew import run_digest
from bot.handlers import send_digest_to_admin
from config import ADMIN_CHAT_ID, PIPELINE_HOURS, SOURCES_FILE
from database.models import create_digest, get_recently_seen_topics
from parsers.source_fetcher import fetch_all_sources, filter_ads

logger = logging.getLogger(__name__)


def _friendly_llm_error(exc: Exception) -> str:
    """Convert provider exceptions into admin-friendly messages."""
    raw = str(exc)
    lowered = raw.lower()
    if "usage limits" in lowered or "rate limit" in lowered or "429" in lowered:
        return (
            "❌ Лимит LLM-провайдера исчерпан. "
            "Автодайджест временно остановлен, пока не сменим ключ/модель."
        )
    if "invalid_request_error" in lowered and "api usage limits" in lowered:
        return (
            "❌ Anthropic временно недоступен из-за лимита. "
            "Автодайджест будет восстановлен после переключения на резервную модель."
        )
    return f"❌ Ошибка DigestCrew:\n<code>{raw}</code>"


# ---------------------------------------------------------------------------
# Core pipeline job
# ---------------------------------------------------------------------------

_PIPELINE_TIMEOUT = 600  # 10 minutes max per run


async def run_pipeline_job(bot: Bot) -> None:
    """Execute the full Воронка pipeline end-to-end (with timeout guard)."""
    try:
        await asyncio.wait_for(_run_pipeline(bot), timeout=_PIPELINE_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error("Pipeline timed out after %d seconds — run killed", _PIPELINE_TIMEOUT)
        try:
            await bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text="⚠️ Пайплайн завис и был принудительно остановлен (таймаут 10 мин).",
            )
        except Exception:
            pass


async def _run_pipeline(bot: Bot) -> None:
    """Actual pipeline logic."""
    logger.info("Pipeline started (Воронка architecture)")

    # 1. Fetch from all sources (Telegram + RSS + Reddit)
    raw_items = await fetch_all_sources(SOURCES_FILE)
    if not raw_items:
        logger.warning("No news items fetched — skipping this run")
        await bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text="⚠️ Пайплайн: не удалось получить новости ни из одного источника.",
        )
        return

    logger.info("Fetched %d raw items total", len(raw_items))

    # 2. Apply hard filter
    news_items = filter_ads(raw_items)
    if not news_items:
        logger.warning("All items were filtered out — skipping this run")
        await bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=(
                f"⚠️ Пайплайн: собрали {len(raw_items)} постов, "
                "но все были отфильтрованы жёстким фильтром."
            ),
        )
        return

    logger.info("After filter_ads: %d items remain", len(news_items))

    # 3. Load recently seen topics for deduplication (last 3 days)
    seen_topics = get_recently_seen_topics(days=3)
    logger.info("Dedup: %d recently seen topics loaded", len(seen_topics))

    # 4. Run DigestCrew (Analyst)
    try:
        digest_result = await run_digest(news_items, seen_topics=seen_topics)
    except Exception as exc:
        logger.exception("DigestCrew failed: %s", exc)
        await bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=_friendly_llm_error(exc),
            parse_mode="HTML",
        )
        return

    all_topics = digest_result.all_topics
    if not all_topics:
        logger.error("DigestCrew produced no topics — aborting")
        await bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text="⚠️ Аналитик не нашёл достойных тем. Проверь логи.",
        )
        return

    logger.info(
        "DigestCrew: %d hard + %d production + %d useful = %d topics total",
        len(digest_result.hard_news),
        len(digest_result.production_cases),
        len(digest_result.useful),
        len(all_topics),
    )

    # 5. Save digest to DB
    topics_as_dicts = [
        {
            "index":      t.index,
            "title":      t.title,
            "summary":    t.summary,
            "category":   t.category,
            "url":        t.url,
            "raw_text":   t.raw_text,
            "has_media":   t.has_media,
            "media_path":  t.media_path,
            "media_paths": getattr(t, "media_paths", [t.media_path] if t.media_path else []),
        }
        for t in all_topics
    ]
    digest_id = create_digest(
        date=digest_result.date,
        topics=topics_as_dicts,
    )
    logger.info("Saved digest #%s (%d topics)", digest_id, len(topics_as_dicts))

    # 6. Send digest menu to admin
    await send_digest_to_admin(bot, digest_id)
    logger.info("Pipeline finished — digest #%s sent to admin", digest_id)


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def setup_scheduler(bot: Bot) -> AsyncIOScheduler:
    """
    Create and configure the AsyncIO scheduler.
    Returns the scheduler (call .start() in main.py).
    """
    scheduler = AsyncIOScheduler(timezone="Europe/Moscow")

    hours_str = ",".join(str(h) for h in PIPELINE_HOURS)
    scheduler.add_job(
        run_pipeline_job,
        trigger="cron",
        hour=hours_str,
        minute=0,
        kwargs={"bot": bot},
        id="daily_pipeline",
        replace_existing=True,
        misfire_grace_time=300,
        coalesce=True,       # если пропустили несколько запусков — выполнить только один
        max_instances=1,     # явно: не запускать параллельно
    )

    logger.info(
        "Scheduler configured: runs at hours %s Moscow time", PIPELINE_HOURS
    )
    return scheduler
