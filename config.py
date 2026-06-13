import os
from dotenv import load_dotenv

load_dotenv(override=True)   # always prefer values from .env over system env vars

# --- LLM ---
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OFOXAI_API_KEY: str = os.getenv("OFOXAI_API_KEY", "")
OFOXAI_BASE_URL: str = os.getenv("OFOXAI_BASE_URL", "https://api.ofox.ai/v1")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "deepseek-chat-v3.1")
LLM_DAILY_BUDGET_USD: float = float(os.getenv("LLM_DAILY_BUDGET_USD", "0.5"))

# --- Telegram Bot (aiogram) ---
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
ADMIN_CHAT_ID: int = int(os.getenv("ADMIN_CHAT_ID", "0"))
TARGET_CHANNEL_ID: str = os.getenv("TARGET_CHANNEL_ID", "")

# --- Telegram Userbot (Telethon) ---
TELEGRAM_API_ID: str   = os.getenv("TELEGRAM_API_ID", "")
TELEGRAM_API_HASH: str = os.getenv("TELEGRAM_API_HASH", "")
TELEGRAM_PHONE: str    = os.getenv("TELEGRAM_PHONE", "")

# Session file saved in the project root; reused on every restart
USERBOT_SESSION: str = os.getenv("USERBOT_SESSION", "userbot.session")

# Folder where Telethon downloads media attachments from posts
MEDIA_DIR: str = os.getenv("MEDIA_DIR", "media")

# --- Content language ---
# Language for generated posts. Examples: "Russian", "English", "Spanish"
CONTENT_LANGUAGE: str = os.getenv("CONTENT_LANGUAGE", "Russian")

# --- Source files (change paths here if needed) ---
SOURCES_FILE: str = "sources_telegram_channels.txt"
STYLE_FILE: str = "my_true_voise.html"

# --- How far back to look across ALL sources (hours) ---
FETCH_CUTOFF_HOURS: int = 48

# --- RSS feeds to monitor ---
RSS_SOURCES: list[str] = [
    "https://huggingface.co/blog/feed.xml",
    "https://openai.com/news/rss.xml",           # OpenAI News (canonical feed)
    "https://openai.com/blog/rss.xml",            # OpenAI Blog (kept as fallback)
    "https://www.anthropic.com/rss.xml",
    "https://blog.google/technology/ai/rss/",
    "https://www.deepmind.com/blog/rss.xml",
    "https://www.marktechpost.com/feed/",         # MarkTechPost AI section
    "https://venturebeat.com/category/ai/feed/",  # VentureBeat AI
    "https://zernel.github.io/huggingface-trending-feed/feed.xml",  # HF Trending (unofficial)
]

# --- Reddit subreddits to fetch via /new.json (no auth required) ---
REDDIT_SUBREDDITS: list[str] = [
    "LocalLLaMA",        # локальные LLM, новые модели, бенчмарки
    "singularity",       # крупные тренды и новости AI
    "comfyui",           # ноды, воркфлоу, FaceSwap, апскейл
    "aivideo",           # Kling, Runway, Sora, Seedance и др.
    "StableDiffusion",   # open-source генерация изображений
    "MachineLearning",   # серьёзный ML/AI ресёрч и релизы
]

# --- Web digest sites to scrape (HTML) ---
# The Rundown AI, TLDR AI, Ben's Bites — ежедневные дайджесты
WEB_DIGEST_SOURCES: list[str] = [
    "https://therundown.ai",
    "https://tldr.tech/ai",
    "https://bensbites.com",
]

# --- Product Hunt AI topics (новые AI-инструменты) ---
PRODUCT_HUNT_URL: str = "https://www.producthunt.com/topics/artificial-intelligence"

# --- Database ---
DATABASE_PATH: str = "posts.db"

# --- Scheduler: hours of day to run the pipeline (24h format) ---
PIPELINE_HOURS: list[int] = [9, 18]

# --- Pipeline settings ---
MAX_SOURCE_POSTS_PER_CHANNEL: int = 5   # posts to fetch per channel
MAX_CHANNELS_PER_RUN: int = 20          # max channels to fetch per pipeline run
POST_VARIANTS_COUNT: int = 1            # WriterCrew generates 1 final post

# Set to "false" to disable the optional VisualPromptDesigner agent
ENABLE_VISUAL_PROMPT: bool = os.getenv("ENABLE_VISUAL_PROMPT", "true").lower() == "true"
