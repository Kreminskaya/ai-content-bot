"""
DigestCrew — Воронка: Сбор → Жёсткий фильтр → Аналитика → Дайджест.

Агент Аналитик принимает очищенный массив постов и формирует JSON-дайджест:
  • Топ-4 Хард-Новостей  (Визуал и Продакшен / Мозги и LLM)
  • Топ-3 Кейсов         (Кейс-истории продакшена: кино/реклама/клипы/игры)
  • Топ-4 Полезностей    (Полезные сервисы / Роботы и железо)

Итого до 11 тем → до 11 кнопок в Telegram для администратора.

Поддерживает дедупликацию: принимает список уже показанных тем (url + заголовок)
из предыдущих дайджестов и передаёт их аналитику как «не повторять».

Используется в scheduler.py.
Когда администратор нажимает кнопку темы, вызывается WriterCrew (agents/crew.py).
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import partial

from crewai import Agent, Crew, LLM, Process, Task

from config import ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL_NAME, LLM_PROVIDER, OFOXAI_API_KEY, OFOXAI_BASE_URL, OPENAI_API_KEY, OPENROUTER_API_KEY

logger = logging.getLogger(__name__)

if ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

_LITELLM_MODEL = f"openai/{LLM_MODEL_NAME}"


def _llm_api_key() -> str:
    if LLM_PROVIDER == "openrouter":
        return OPENROUTER_API_KEY
    if LLM_PROVIDER == "ofoxai":
        return OFOXAI_API_KEY
    if LLM_PROVIDER == "deepseek":
        return DEEPSEEK_API_KEY
    return OPENAI_API_KEY if LLM_PROVIDER == "openai" else ""


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _llm(temperature: float = 0.3) -> LLM:
    kwargs = {"temperature": temperature}
    if LLM_PROVIDER == "openrouter":
        kwargs["api_base"] = "https://openrouter.ai/api/v1"
    elif LLM_PROVIDER == "ofoxai":
        kwargs["api_base"] = OFOXAI_BASE_URL
    elif LLM_PROVIDER == "deepseek":
        kwargs["api_base"] = DEEPSEEK_BASE_URL
    return LLM(model=_LITELLM_MODEL, api_key=_llm_api_key(), **kwargs)


# ---------------------------------------------------------------------------
# Category constants (shared with bot/handlers.py via import)
# ---------------------------------------------------------------------------

CATEGORY_VISUAL      = "Визуал и Продакшен"
CATEGORY_LLM         = "Мозги и LLM"
CATEGORY_SERVICES    = "Полезные сервисы"
CATEGORY_ROBOTICS    = "Роботы и железо"
CATEGORY_PRODUCTION  = "Кейс-истории продакшена"

# CATEGORY_PROMPTS removed — rerouted to services
CATEGORY_PROMPTS     = "Полезные сервисы"

HARD_CATEGORIES      = {CATEGORY_VISUAL, CATEGORY_LLM}
USEFUL_CATEGORIES    = {CATEGORY_SERVICES, CATEGORY_ROBOTICS}
PRODUCTION_CATEGORY  = {CATEGORY_PRODUCTION}

ALL_CATEGORIES = (
    HARD_CATEGORIES | USEFUL_CATEGORIES | PRODUCTION_CATEGORY
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class DigestTopic:
    """One selected topic from the analyst's digest."""
    index: int            # 1-N, used as button index in Telegram
    title: str            # short headline
    summary: str          # 2-3 sentence description
    category: str         # one of the five categories
    url: str              # source permalink
    raw_text: str = ""    # original post text (for WriterCrew context)
    has_media: bool = False  # whether the original post had media
    media_path: str = ""     # path to pre-downloaded media file (if available)


@dataclass
class DigestResult:
    """Full digest output from the Analyst agent."""
    date: str
    hard_news:        list[DigestTopic] = field(default_factory=list)  # max 4 (cat 1&2)
    production_cases: list[DigestTopic] = field(default_factory=list)  # max 3 (cat 5)
    useful:           list[DigestTopic] = field(default_factory=list)  # max 4 (cat 3&4)

    @property
    def all_topics(self) -> list[DigestTopic]:
        """
        All topics in order: hard news → production cases → useful.
        Indices are assigned sequentially 1..N here.
        """
        return self.hard_news + self.production_cases + self.useful

    def get_topic(self, index: int) -> DigestTopic | None:
        for t in self.all_topics:
            if t.index == index:
                return t
        return None


# ---------------------------------------------------------------------------
# Analyst agent
# ---------------------------------------------------------------------------

_ANALYST_SYSTEM = """\
Ты — Главред AI-издания с широким взглядом: следишь и за технологическими новостями, и за \
реальными кейсами применения ИИ в индустрии.

Тебе на вход поступает очищенный массив сырых постов из Telegram-каналов, Reddit, RSS-лент \
и дайджест-сайтов.

Твоя задача — отбросить инфошум (суды, падение акций, корпоративные скандалы, \
пресс-релизы без конкретики) и распределить релевантный контент по 5 категориям:

1. Визуал и Продакшен
   • ComfyUI, FaceSwap, Image/Video generation, ноды, воркфлоу, Figma+AI
   • Новые инструменты для генерации фото и видео

2. Мозги и LLM
   • Релизы новых моделей: OpenAI, Anthropic, Google, DeepSeek, Qwen, локальные LLM
   • Бенчмарки, исследования, RLHF, рассуждение, агенты

3. Полезные сервисы
   • Новые AI-инструменты для работы, агрегаторы, платформы
   • Product Hunt AI — новинки из категории AI Tools
   • TLDR AI, The Rundown, Ben's Bites — свежие тулзы

4. Роботы и железо
   • Гуманоидные роботы, AI-дроны, робособаки — новые модели и демо
   • Китайские, японские, американские разработки (Unitree, Boston Dynamics, Tesla Optimus и др.)
   • AI-чипы, нейроморфные процессоры, новое железо для инференса
   • HuggingFace: популярные новые модели, датасеты, trending repos
   • GitHub: интересные AI-репо (агенты, vibe-coding, локальные LLM)

5. Кейс-истории продакшена
   • Режиссёры, студии, бренды используют ИИ для съёмок (реклама, клипы, мини-сериалы, \
трейлеры, short films)
   • Кейсы «сняли рекламу полностью на ИИ», «музыкальный клип на AI», «промо-ролик на Runway»
   • Крупные продакшены применяют ComfyUI / Runway / Kling / Gen-2 / Seedance в реальных \
проектах
   • Ключевые слова-маркеры: "commercial", "music video", "short film", "ad", "campaign", \
"shot with AI", "made with AI", "AI-generated video", "AI-generated film", "AI-generated ad"

ВАЖНЫЕ ПРАВИЛА ОТБОРА:
✅ Официальные релизы, которые можно потрогать прямо сейчас
✅ Яркие кейсы применения ИИ с реальным результатом
✅ Практичный контент, который читатель может использовать сегодня
✅ Из нескольких источников об одном событии — выбирай лучший, остальные отбрасывай
✅ СВЕЖЕСТЬ: предпочитай материалы ≤ 48 часов. При прочих равных бери более свежее.
   Не включай темы старше 3 дней, если в ленте есть свежие альтернативы.

❌ Суды, штрафы, регуляторные скандалы
❌ Корпоративные заявления без технической конкретики
❌ Темы без ссылки на реальный материал
❌ Чужие платные курсы, мастер-классы, вебинары, онлайн-школы — это реклама чужого
   обучения, не наш контент. Пропускай любые анонсы вида «запишись на курс X»,
   «мастер-класс по Y», «вебинар Z», «открыт набор в...», «обучение по...».
❌ Объявления о запуске чужих образовательных продуктов или потоков обучения.
❌ Посты-подборки "топ промптов", "полезные промпты ChatGPT", "лайфхаки с ИИ" — это не наш контент.
"""


def _make_analyst(llm: LLM) -> Agent:
    return Agent(
        role="Главред AI-дайджеста",
        goal=(
            "Проанализировать очищенную ленту постов и сформировать структурированный "
            "JSON-дайджест: 4 хард-новости, до 3 кейсов продакшена, 4 полезности."
        ),
        backstory=_ANALYST_SYSTEM,
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# Analyst task
# ---------------------------------------------------------------------------

def _analysis_task(
    agent: Agent,
    news_feed: str,
    seen_block: str,
) -> Task:
    dedup_section = (
        f"\n\n─── ТЕМЫ, КОТОРЫЕ УЖЕ БЫЛИ В ПРЕДЫДУЩИХ ДАЙДЖЕСТАХ (НЕ ПОВТОРЯЙ) ───\n"
        f"{seen_block}\n"
        if seen_block.strip()
        else ""
    )

    return Task(
        description=(
            "Проанализируй ленту постов ниже и сформируй дайджест.\n\n"
            "ОБЯЗАТЕЛЬНЫЙ ФОРМАТ ОТВЕТА — строго JSON (и только JSON, без пояснений вокруг):\n\n"
            "{\n"
            '  "hard_news": [\n'
            "    {\n"
            '      "title": "Заголовок темы одной строкой",\n'
            '      "summary": "Краткая суть в 2-3 предложениях.",\n'
            '      "category": "Визуал и Продакшен",\n'
            '      "url": "https://..."\n'
            "    }\n"
            "  ],\n"
            '  "production_cases": [\n'
            "    {\n"
            '      "title": "Бренд/студия + что сделали на ИИ",\n'
            '      "summary": "Кейс: кто, что, какой инструмент, результат.",\n'
            '      "category": "Кейс-истории продакшена",\n'
            '      "url": "https://..."\n'
            "    }\n"
            "  ],\n"
            '  "useful": [\n'
            "    {\n"
            '      "title": "Заголовок инструмента/лайфхака",\n'
            '      "summary": "Краткая суть в 2-3 предложениях.",\n'
            '      "category": "Полезные сервисы",\n'
            '      "url": "https://..."\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "ПРАВИЛА ДЛЯ ЗАПОЛНЕНИЯ МАССИВОВ:\n"
            "• title — КОНКРЕТНЫЙ и ИНФОРМАТИВНЫЙ. Формат: '[Что/Кто] — [что произошло]'.\n"
            "  ✅ 'Runway выпустил Act-One — генерация мимики из видео'\n"
            "  ✅ 'Claude 4 доступен в API — контекст 1M токенов'\n"
            "  ❌ 'Интересный релиз в мире видео'\n"
            "  ❌ 'Новая модель от Anthropic'\n"
            "  Заголовок должен быть понятен БЕЗ чтения summary.\n"
            "• hard_news         — 4 объекта (ТОЛЬКО категории: 'Визуал и Продакшен' или 'Мозги и LLM')\n"
            "• production_cases  — 0–3 объекта (ТОЛЬКО категория: 'Кейс-истории продакшена')\n"
            "  Если таких кейсов в ленте нет — передай пустой массив []\n"
            "• useful            — 4 объекта (ТОЛЬКО категории: 'Полезные сервисы' или 'Роботы и железо')\n"
            "• Если подходящих постов меньше нужного — включи лучшее что есть\n"
            "• url должен быть реальной ссылкой из ленты, не придумывай\n"
            "• Сортировка внутри каждого массива: сначала официальные релизы крупных игроков "
            "и яркие кейсы, потом менее важные\n"
            "• Свежесть — важный критерий: предпочитай темы ≤ 48 часов. Не бери темы\n"
            "  старше 3 дней, если есть свежие альтернативы того же качества.\n"
            "• НЕЛЬЗЯ включать: чужие курсы, мастер-классы, вебинары, наборы на обучение —\n"
            "  любые анонсы образовательных продуктов других авторов/школ.\n"
            "• Отвечай ТОЛЬКО валидным JSON — никакого текста до или после\n"
            f"{dedup_section}"
            "─────────────── ЛЕНТА ПОСТОВ ───────────────\n\n"
            f"{news_feed}"
        ),
        expected_output=(
            "Валидный JSON-объект с тремя массивами: "
            "hard_news (4), production_cases (0-3), useful (4). "
            "Только JSON, никакого текста вокруг."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# JSON output parser
# ---------------------------------------------------------------------------

def _parse_digest_json(
    raw: str,
    news_items: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Extract hard_news, production_cases, useful from the analyst's output.
    Returns (hard_news, production_cases, useful) as lists of dicts.
    Graceful fallback on malformed JSON.
    """
    # Strip markdown code fences that LLMs sometimes add despite instructions
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        logger.error("Analyst output contains no JSON block. Raw: %s", raw[:300])
        return [], [], []

    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError as exc:
        logger.error("JSON decode failed: %s. Raw snippet: %s", exc, raw[:300])
        return [], [], []

    hard_news        = data.get("hard_news",        [])
    production_cases = data.get("production_cases", [])
    useful           = data.get("useful",           [])

    # Enrich with raw_text AND has_media for WriterCrew context.
    # LLM often writes a product URL (e.g. huggingface.co) instead of the
    # source t.me post URL — so we do URL lookup first, then keyword fallback.
    url_to_item: dict[str, dict] = {
        item.get("url", ""): item for item in news_items
    }

    def _find_source(topic: dict) -> dict:
        """Return the best-matching news_item for a digest topic."""
        # 1. Exact URL match
        src = url_to_item.get(topic.get("url", ""), {})
        if src:
            return src
        # 2. Keyword fallback: score items by shared title words
        title_words = set(topic.get("title", "").lower().split())
        # Remove noise words
        stop = {"—", "-", "и", "в", "с", "на", "для", "по", "от", "из", "или"}
        title_words -= stop
        if len(title_words) < 2:
            return {}
        best, best_score = {}, 0
        for ni in news_items:
            text_words = set(ni.get("text", "").lower().split())
            score = len(title_words & text_words)
            if score > best_score:
                best_score, best = score, ni
        if best_score >= 2:
            # Override LLM URL with actual t.me source URL
            topic["url"] = best.get("url", topic.get("url", ""))
            return best
        return {}

    for item in hard_news + production_cases + useful:
        src = _find_source(item)
        item["raw_text"] = src.get("text", "")
        item["has_media"] = src.get("has_media", False) or bool(src.get("media_path"))
        item["media_path"] = src.get("media_path", "") or ""  # carry through pre-downloaded file

    return hard_news, production_cases, useful


# ---------------------------------------------------------------------------
# Synchronous runner (thread executor)
# ---------------------------------------------------------------------------

def _run_digest_sync(
    news_items: list[dict],
    seen_topics: list[dict],
) -> DigestResult:
    from datetime import datetime  # noqa: PLC0415

    if not news_items:
        raise ValueError("news_items is empty — nothing to analyse")

    def _fmt_item(i: int, item: dict) -> str:
        date_str = str(item.get("date", "?"))[:25]
        has_media = item.get("media_path") or item.get("has_media")
        media_note = " [медиа]" if has_media else ""
        return (
            f"[{i+1}] {item.get('source', '')}\n"
            f"Дата: {date_str}\n"
            f"URL: {item.get('url', '')}{media_note}\n\n"
            f"{item.get('text', '')[:600]}"
        )

    news_feed = "\n\n".join(
        _fmt_item(i, item) for i, item in enumerate(news_items[:60])
    )

    # ── Stage 1: Temporal Context Injection ────────────────────────────────
    from datetime import datetime as _dt
    _today = _dt.now().strftime("%d %B %Y")  # e.g. "17 April 2026"
    _temporal_header = (
        f"══ ТЕКУЩАЯ ДАТА: {_today} ══\n"
        "Все материалы ниже — свежие. Оценивай их актуальность "
        "относительно этой даты.\n\n"
    )
    news_feed = _temporal_header + news_feed
    # ───────────────────────────────────────────────────────────────────────

    # Build dedup block for the analyst prompt
    seen_lines: list[str] = []
    for t in seen_topics[:30]:
        title = t.get("title", "")
        url   = t.get("url", "")
        if title:
            seen_lines.append(f"• {title}  [{url}]" if url else f"• {title}")
    seen_block = "\n".join(seen_lines)

    llm     = _llm()
    analyst = _make_analyst(llm)
    a_task  = _analysis_task(analyst, news_feed, seen_block)

    crew = Crew(
        agents=[analyst],
        tasks=[a_task],
        process=Process.sequential,
        verbose=False,
    )
    crew.kickoff()

    raw_output = str(a_task.output) if a_task.output else ""
    hard_raw, production_raw, useful_raw = _parse_digest_json(raw_output, news_items)

    # Convert to DigestTopic objects with sequential indices
    idx = 1

    hard_topics: list[DigestTopic] = []
    for item in hard_raw[:4]:
        hard_topics.append(DigestTopic(
            index=idx,
            title=item.get("title", f"Тема {idx}"),
            summary=item.get("summary", ""),
            category=item.get("category", CATEGORY_LLM),
            url=item.get("url", ""),
            raw_text=item.get("raw_text", ""),
            has_media=item.get("has_media", False),
            media_path=item.get("media_path", "") or "",
        ))
        idx += 1

    production_topics: list[DigestTopic] = []
    for item in production_raw[:3]:
        production_topics.append(DigestTopic(
            index=idx,
            title=item.get("title", f"Тема {idx}"),
            summary=item.get("summary", ""),
            category=CATEGORY_PRODUCTION,
            url=item.get("url", ""),
            raw_text=item.get("raw_text", ""),
            has_media=item.get("has_media", False),
            media_path=item.get("media_path", "") or "",
        ))
        idx += 1

    useful_topics: list[DigestTopic] = []
    for item in useful_raw[:4]:
        useful_topics.append(DigestTopic(
            index=idx,
            title=item.get("title", f"Тема {idx}"),
            summary=item.get("summary", ""),
            category=item.get("category", CATEGORY_SERVICES),
            url=item.get("url", ""),
            raw_text=item.get("raw_text", ""),
            has_media=item.get("has_media", False),
            media_path=item.get("media_path", "") or "",
        ))
        idx += 1

    date_str = datetime.now().strftime("%d.%m.%Y")
    logger.info(
        "DigestCrew finished: %d hard / %d production / %d useful",
        len(hard_topics), len(production_topics), len(useful_topics),
    )
    return DigestResult(
        date=date_str,
        hard_news=hard_topics,
        production_cases=production_topics,
        useful=useful_topics,
    )


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def run_digest(
    news_items: list[dict],
    seen_topics: list[dict] | None = None,
) -> DigestResult:
    """
    Run the DigestCrew Analyst in a thread executor.

    Args:
        news_items:   filtered post list from fetch_all_sources + filter_ads
        seen_topics:  list of dicts {title, url} from recent digests (for dedup)

    Returns DigestResult with up to 11 topics (4 hard + 3 production + 4 useful).
    """
    logger.info(
        "DigestCrew: analysing %d posts (seen_topics=%d)",
        len(news_items),
        len(seen_topics) if seen_topics else 0,
    )
    loop = asyncio.get_event_loop()
    result: DigestResult = await loop.run_in_executor(
        None,
        partial(_run_digest_sync, news_items, seen_topics or []),
    )
    return result
