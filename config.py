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

# --- Interface & content language ---
# Initial default language for the whole bot (UI + generated content).
# Accepts "ru"/"en" or "Russian"/"English". Users switch it live via /language;
# the choice is remembered in runtime_state.json. See bot/i18n.py.
LANGUAGE: str = os.getenv("LANGUAGE", os.getenv("CONTENT_LANGUAGE", "ru"))
# Legacy alias, still read by older code paths.
CONTENT_LANGUAGE: str = os.getenv("CONTENT_LANGUAGE", LANGUAGE)

# --- Category tags (stable internal identifiers, NOT translated) ---
# Enum-like routing keys: the analyst emits them, handlers route on them.
# The user never sees the raw tag — only an emoji + the localized section header
# (bot/i18n.py) + the topic title (written in the chosen language).
# English tags (used when the analyst runs in English)
CATEGORY_VISUAL     = "Visual & Production"
CATEGORY_LLM        = "AI & LLM"
CATEGORY_SERVICES   = "Useful Tools"
CATEGORY_ROBOTICS   = "Robots & Hardware"
CATEGORY_PRODUCTION = "Production Cases"

# Russian tags (used when the analyst runs in Russian) — original taxonomy.
CATEGORY_VISUAL_RU     = "Визуал и Продакшен"
CATEGORY_LLM_RU        = "Мозги и LLM"
CATEGORY_SERVICES_RU   = "Полезные сервисы"
CATEGORY_ROBOTICS_RU   = "Роботы и железо"
CATEGORY_PRODUCTION_RU = "Кейс-истории продакшена"

# Groups are language-agnostic UNIONS: handlers route correctly whether the
# analyst emitted Russian or English category tags.
HARD_CATEGORIES     = {CATEGORY_VISUAL, CATEGORY_LLM, CATEGORY_VISUAL_RU, CATEGORY_LLM_RU}
USEFUL_CATEGORIES   = {CATEGORY_SERVICES, CATEGORY_ROBOTICS, CATEGORY_SERVICES_RU, CATEGORY_ROBOTICS_RU}
PRODUCTION_CATEGORY = {CATEGORY_PRODUCTION, CATEGORY_PRODUCTION_RU}
ALL_CATEGORIES      = HARD_CATEGORIES | USEFUL_CATEGORIES | PRODUCTION_CATEGORY

# --- Source files (change paths here if needed) ---
SOURCES_FILE: str = "sources_telegram_channels.txt"
STYLE_FILE: str = "my_true_voise.html"

# --- How far back to look across ALL sources (hours) ---
FETCH_CUTOFF_HOURS: int = 48

# --- RSS feeds to monitor ---
# Add any RSS feed URLs relevant to your niche.
# Examples for AI: https://openai.com/news/rss.xml, https://www.anthropic.com/rss.xml
# Examples for fashion: https://www.vogue.com/feed/rss, https://hypebeast.com/feed
RSS_SOURCES: list[str] = [
    # Add your RSS feeds here
]

# --- Reddit subreddits to fetch via /new.json (no auth required) ---
# Add subreddit names (without r/) relevant to your niche.
REDDIT_SUBREDDITS: list[str] = [
    # Add your subreddits here, e.g.: "MachineLearning", "fashion", "photography"
]

# --- Web digest sites to scrape (HTML) ---
# Add newsletter/digest sites to scrape for headlines.
WEB_DIGEST_SOURCES: list[str] = [
    # Add digest sites here, e.g.: "https://tldr.tech/ai"
]

# --- Product Hunt (optional) ---
# Set to "" to disable Product Hunt fetching entirely
PRODUCT_HUNT_URL: str = os.getenv("PRODUCT_HUNT_URL", "")

# --- GitHub trending repos (optional) ---
# Leave empty to disable. Add topic tags relevant to your niche.
# Examples for AI: "machine-learning,llm,ai-agent"
# Examples for fashion: "fashion,style,clothing"
_gh_topics_env = os.getenv("GITHUB_TRENDING_TOPICS", "")
GITHUB_TRENDING_TOPICS: list[str] = [t.strip() for t in _gh_topics_env.split(",") if t.strip()]

# --- HuggingFace papers + models (optional) ---
# Set to 0 to disable
HUGGINGFACE_DAILY_PAPERS_LIMIT: int = int(os.getenv("HUGGINGFACE_DAILY_PAPERS_LIMIT", "0"))
HUGGINGFACE_TRENDING_MODELS_LIMIT: int = int(os.getenv("HUGGINGFACE_TRENDING_MODELS_LIMIT", "0"))

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
