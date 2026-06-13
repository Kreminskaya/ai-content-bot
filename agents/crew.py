"""
WriterCrew — CrewAI pipeline: Researcher → Writer → Editor → RhythmChecker → Formatter → VisualDesigner.

Pipeline triggered when admin clicks a digest topic button (or via /post manual flow).

Input:
  news_items : list[dict] — each has source, url, text, date, media_path, has_media
  recent_posts : str — last 5-10 published posts (separated by "\n---\n")
  user_reviews : str — community/user feedback on the topic (optional)

Output:
  PipelineResult with ONE final post text + ONE image-generation prompt.

ARCHITECTURE NOTES (v2 rewrite):
  • All agent backstories are SHORT (5-15 lines) — DeepSeek V4 Flash cannot hold
    150-line backstories + 80-line task descriptions + 80 regex patterns simultaneously.
  • No regex post-processing for "AI patterns" — fix the prompt, not the output.
  • Researcher outputs structured FACT BLOCK. Writer receives facts as numbered list.
  • Editor cross-checks writer's draft against the FACT BLOCK (not the original text —
    the original text is too noisy; the Researcher's extraction is the ground truth).
  • RhythmChecker checks against recent_posts context for pattern diversity.
  • Formatter applies SKILL.md rules for visual formatting (emojis, bold, blockquotes).
  • VisualDesigner generates image prompt (optional, controlled by ENABLE_VISUAL_PROMPT).
"""

import asyncio
import logging
import os
import random
import re
from dataclasses import dataclass, field
from functools import partial

from crewai import Agent, Crew, LLM, Process, Task, TaskOutput

from config import (
    ANTHROPIC_API_KEY,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    ENABLE_VISUAL_PROMPT,
    LLM_MODEL_NAME,
    LLM_PROVIDER,
    OFOXAI_API_KEY,
    OFOXAI_BASE_URL,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
)

logger = logging.getLogger(__name__)

if ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

_LITELLM_MODEL = f"openai/{LLM_MODEL_NAME}"


def _llm_key() -> str:
    if LLM_PROVIDER == "openrouter":
        return OPENROUTER_API_KEY
    if LLM_PROVIDER == "ofoxai":
        return OFOXAI_API_KEY
    if LLM_PROVIDER == "deepseek":
        return DEEPSEEK_API_KEY
    return OPENAI_API_KEY if LLM_PROVIDER == "openai" else ANTHROPIC_API_KEY


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Everything the WriterCrew produces in one run."""
    variants: list[str] = field(default_factory=list)   # always 1 element
    image_prompt: str | None = None
    researcher_summary: str = ""


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _llm(temperature: float = 0.7) -> LLM:
    kwargs = {"temperature": temperature}
    if LLM_PROVIDER == "openrouter":
        kwargs["api_base"] = "https://openrouter.ai/api/v1"
    elif LLM_PROVIDER == "ofoxai":
        kwargs["api_base"] = OFOXAI_BASE_URL
    elif LLM_PROVIDER == "deepseek":
        kwargs["api_base"] = DEEPSEEK_BASE_URL
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        kwargs["reasoning_effort"] = "medium"
    return LLM(model=_LITELLM_MODEL, api_key=_llm_key(), **kwargs)


# ---------------------------------------------------------------------------
# POST TYPE TAXONOMY — pre-generation diversity
# ---------------------------------------------------------------------------

POST_TYPE_TAXONOMY = {
    "hot_take": {
        "name": "🔥 Hot Take",
        "description": (
            "Сильное мнение по новости. Структура: Смелое утверждение → "
            "Почему так думаю → Что все упускают."
        )
    },
    "field_notes": {
        "name": "📝 Заметки из поля",
        "description": (
            "Личная реакция, как будто только что прочитала что-то интересное. "
            "Структура: Что удивило → Почему → О чём теперь думаю."
        )
    },
    "technical_explainer": {
        "name": "🔧 Разбор",
        "description": (
            "Как это реально работает? Без маркетинга. "
            "Структура: Убери жаргон → Механизм → Почему это важно технически."
        )
    },
    "so_what": {
        "name": "💡 И что?",
        "description": (
            "Новость как точка данных в более крупном паттерне. "
            "Структура: Факт → Контекст → Большая история."
        )
    },
    "contrarian": {
        "name": "⚡ Против течения",
        "description": (
            "Позиция, отличная от пресс-релиза — только если есть основания. "
            "Структура: Что говорят → Что вижу на практике → Какие вопросы остаются."
        )
    },
    "analogy": {
        "name": "🎭 Аналогия",
        "description": (
            "Объяснить AI-разработку через аналогию из другой области. "
            "Структура: Аналогия → Карта на AI → Следствие."
        )
    },
    "comparison": {
        "name": "⚖️ Сравнение",
        "description": (
            "Это vs. то — что реально отличается? "
            "Структура: Сравнение → Ключевое отличие → Почему это важно."
        )
    },
    "practical_guide": {
        "name": "🛠️ Практика",
        "description": (
            "Как это использовать. Конкретные шаги, промпты, примеры. "
            "Структура: Задача → Инструмент/подход → Пошагово."
        )
    },
}


def _select_post_type(recent_types: list[str] | None = None) -> tuple[str, str]:
    """Select a post type that wasn't used in the last 5 posts."""
    all_keys = list(POST_TYPE_TAXONOMY.keys())
    if recent_types:
        recent_set = set(recent_types[-5:])
        available = [k for k in all_keys if k not in recent_set]
        if not available:
            available = all_keys
    else:
        available = all_keys
    key = random.choice(available)
    pt = POST_TYPE_TAXONOMY[key]
    return pt["name"], pt["description"]


# ---------------------------------------------------------------------------
# Agent builders — SHORT backstories (5-15 lines max)
# DeepSeek V4 Flash can't hold 150-line backstories. Keep it tight.
# ---------------------------------------------------------------------------

def _make_researcher(llm: LLM) -> Agent:
    return Agent(
        role="Fact Extractor — AI News Researcher",
        goal=(
            "Извлечь из исходного текста ВСЕ конкретные факты: цифры, названия моделей, "
            "версии, даты, цены, возможности, ограничения, ссылки. "
            "Вернуть структурированный FACT BLOCK. Никаких обобщений, никакой оценки."
        ),
        backstory=(
            "Ты экстрактор фактов. Твоя работа — вытащить из сырого текста всё твёрдое: "
            "цифры, названия, даты, URL. Ты не понимаешь новость — ты её препарируешь. "
            "Каждый факт в FACT BLOCK должен быть проверяемым: содержит цифру, название, "
            "или характеристику из текста. "
            "URL — КРИТИЧЕН. Если в исходнике есть ссылка — она ДОЛЖНА быть в FACT BLOCK. "
            "Если ссылки нет — пиши 'URL: отсутствует'. Не гадай, не выдумывай. "
            "Ты не журналист, не аналитик, не писатель. Ты скальпель."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def _make_writer(llm: LLM) -> Agent:
    """Writer uses temperature=0.5 — balance of facts and voice."""
    return Agent(
        role="Редактор новостного AI-канала",
        goal=(
            "Написать пост, который содержит ВСЕ факты из FACT BLOCK. "
            "Ни один факт не теряется. Ничего не выдумывать. "
            "Живой голос, не пресс-релиз. КАТЕГОРИЧЕСКИ без первого лица."
        ),
        backstory=(
            "Ты пишешь новостные посты для Telegram-канала об AI.\n\n"
            "ГОЛОС — молодой, живой, как рассказываешь другу про крутую новинку:\n"
            "• Разговорный русский. Не официальный, не переведённый.\n"
            "• Синтаксис простой: подлежащее → сказуемое.\n"
            "• Сленг — обязательно там где уместен: «выкатили», «запустили», «прикрутили»,\n"
            "  «зарелизили», «задеплоили», «набенчмаркали», «сбили бенчи у GPT-4».\n"
            "• Эмоция — ТОЛЬКО фактическая: не «невероятно», а «за 2 секунды вместо 40».\n"
            "• ЗАПРЕЩЕНО добавлять в конце поста мнение, скептицизм или оценку:\n"
            "  — «посмотрим удастся ли», «а получится ли», «пока непонятно»\n"
            "  — «неизвестно взлетит ли», «конкуренты не дремлют», «рынок ответит»\n"
            "  — «остаётся только ждать», «поживём — увидим», «что ж посмотрим»\n"
            "  Концовка поста = последний ФАКТ. Не мнение. Не предсказание.\n\n"
            "ТЕХНИЧЕСКИЕ ДАННЫЕ — ОБЯЗАТЕЛЬНЫ:\n"
            "• Если в FACT BLOCK есть бенчмарки, цифры, миллисекунды, сравнения с другими\n"
            "  моделями — ВСЁ это должно быть в посте. Это и есть главная ценность.\n"
            "• Числа, версии, параметры — дословно из FACT BLOCK, не округляй и не выкидывай.\n"
            "• Сравнения с конкурентами (быстрее GPT-4 в 3x, дешевле Claude на 40%) — СОХРАНЯЙ.\n\n"
            "ТЕРМИНЫ — инженерный вайб, не пересказ для бабушки:\n"
            "• Используй термины сообщества: липсинк, лора, файнтюн, чекпоинт, апскейл,\n"
            "  воркфлоу, инпейнтинг, промпт, инференс, контекстное окно, рэг, агент.\n"
            "• Можно транслит (липсинк) или английский (LoRA) — по ситуации.\n"
            "• НЕЛЬЗЯ заменять термины описанием: «липсинк» ≠ «синхронизация движения губ».\n"
            "• Названия моделей, инструментов, сервисов — ТОЧНО как в источнике.\n\n"
            "ЗАГОЛОВОК — факт, не продажа:\n"
            "• Что произошло — прямо. Без «революционный», «невероятный», «впервые в истории».\n"
            "• ✅ «DeepSeek V3 — 671B params, бьёт GPT-4 на 12 из 15 бенчей, открытые веса»\n"
            "• ❌ «DeepSeek совершил прорыв который изменит всё»\n\n"
            "КАТЕГОРИЧЕСКИЙ ЗАПРЕТ: не писать от первого лица. "
            "Никаких «я», «мне», «мой», «буду», «попробовал», «тестировал». "
            "Ты — голос канала, не персонаж.\n\n"
            "Ссылки вплетай в текст: «выложили на GitHub», «модель на HuggingFace». "
            "Никаких «Ссылка:», «Источник:», «Попробовать:»."
        ),
        skills=["./skills/publishing-rules"],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def _make_editor(llm: LLM) -> Agent:
    return Agent(
        role="Главред — факт-чекер и редактор",
        goal=(
            "Сверить черновик Writer'а с FACT BLOCK от Researcher'а. "
            "Каждый факт из FACT BLOCK должен быть в финальном посте. "
            "Ничего выдуманного. "
            "Голос канала сохранён. Первое лицо — под нож."
        ),
        backstory=(
            "Ты главред AI-блога. У тебя есть FACT BLOCK — эталон фактов. "
            "У тебя есть черновик Writer'а. Твоя работа — построчное сравнение:\n\n"
            "1. Каждый факт из FACTS (FACT BLOCK) присутствует в черновике?\n"
            "2. В черновике нет ничего, чего нет в FACT BLOCK? "
            "(выдуманные метафоры, сценарии, чужой опыт)\n"
            "3. КАТЕГОРИЧЕСКАЯ ЗАПРЕЩЁНКА:\n"
            "   — ПЕРВОЕ ЛИЦО: «я», «мне», «меня», «мой», «буду», «попробовал», «тестировал», «юзаю»\n"
            "   — Выдуманные планы: «буду тестировать», «надеюсь появится», «жду песочницу»\n"
            "   — Ссылки на источники в тексте: «ссылаться на пост-источник», «источник: @канал»\n"
            "   — AI-штампы: «Попробовать:», «Ссылка:», «Источник:», «Подробнее:»\n"
            "   — Бытовые аналогии: кухня, еда, автомобиль\n"
            "   Нашёл — УДАЛИ.\n\n"
            "Ты не переписываешь голос. Ты защищаешь читателя от выдумок и AI-штампов."
        ),
        skills=["./skills/publishing-rules"],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def _make_rhythm_checker(llm: LLM) -> Agent:
    return Agent(
        role="Ритм-редактор",
        goal=(
            "Сравнить пост с последними опубликованными постами канала. "
            "Убедиться, что нет повторяющихся паттернов (тип хука, тональность, "
            "структура). Если есть — минимально скорректировать."
        ),
        backstory=(
            "Ты отвечаешь за то, чтобы канал не звучал монотонно. "
            "Ты читаешь последние посты и проверяешь новый на повторения. "
            "Не переписываешь, а замечаешь паттерны: три подряд одинаковых хука, "
            "два подряд скептических поста, повторяющиеся финалы.\n\n"
            "Если пост уникален по ритму — верни без изменений. "
            "Если есть повторение — минимальная правка: другой хук или финал. "
            "Не трогай факты, не меняй голос."
        ),
        skills=["./skills/publishing-rules"],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def _make_formatter(llm: LLM) -> Agent:
    return Agent(
        role="Оформитель постов",
        goal=(
            "Применить форматные правила из SKILL.md к финальному тексту: "
            "жирный заголовок, 1-3 эмодзи, блокквоты для длинных промптов, "
            "пустые строки между абзацами. Не менять содержание и голос."
        ),
        backstory=(
            "Ты отвечаешь за визуальное оформление поста. Получаешь готовый текст "
            "и делаешь его красивым — не меняя содержание, голос и факты. "
            "Правила форматирования — в твоём SKILL.md файле.\n\n"
            "Твоя мантра: 1-3 эмодзи органично, <b>только заголовок</b>, "
            "blockquote для длинных промптов, пустые строки между абзацами. "
            "Никаких хештегов."
        ),
        skills=["./skills/publishing-rules"],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def _make_visual_designer(llm: LLM) -> Agent:
    return Agent(
        role="Visual Prompt Designer",
        goal=(
            "По финальному тексту поста создать один точный английский промпт "
            "для генерации обложки (Midjourney / DALL·E / Stable Diffusion)."
        ),
        backstory=(
            "Ты специалист по созданию промптов для генерации изображений. "
            "Смотришь на готовый пост, понимаешь его настроение и главную идею, "
            "и придумываешь один визуально конкретный английский промпт (40-80 слов). "
            "Промпт: стиль, атмосфера, объекты, свет, цветовая палитра."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# Task builders — SHORT, structured, focused
# ---------------------------------------------------------------------------

def _research_task(agent: Agent, news_feed: str, user_reviews: str = "") -> Task:
    reviews_section = ""
    if user_reviews.strip():
        reviews_section = (
            "\n\n=== ОТЗЫВЫ ПОЛЬЗОВАТЕЛЕЙ ===\n"
            f"{user_reviews}\n\n"
            "Включи релевантные факты из отзывов в FACT BLOCK, "
            "если они добавляют конкретику (баги, реальный опыт, сравнения)."
        )

    return Task(
        description=(
            "Извлеки ВСЕ конкретные факты из исходного текста ниже. "
            "Верни структурированный FACT BLOCK в таком формате:\n\n"
            "=== FACT BLOCK ===\n"
            "TITLE: [название продукта/модели/события — ОДНА строка]\n"
            "SOURCE: [источник]\n"
            "URL: [ссылка, или 'отсутствует' если ссылки нет]\n"
            "FACTS:\n"
            "1. [конкретный факт с цифрой/названием/датой]\n"
            "2. [конкретный факт]\n"
            "...\n"
            "CAPABILITIES: [что делает/умеет — одной строкой, или 'не указаны']\n"
            "LIMITATIONS: [ограничения, если указаны — иначе 'не указаны']\n"
            "SOURCE_TONE: [нейтральный/восторженный/скептический — одним словом]\n"
            "=== END FACT BLOCK ===\n\n"
            "ЖЁСТКИЕ ПРАВИЛА:\n"
            "• Только то, что есть в тексте. Ничего не придумывай.\n"
            "• Каждый FACT — минимум одна конкретная деталь: цифра, название, дата, цена.\n"
            "• URL: ТОЛЬКО внешняя ссылка на продукт/инструмент/статью (не t.me!). "
            "Ссылки вида t.me/channel/123 — это SOURCE (канал-источник), они НЕ идут в URL. "
            "Приоритет ссылок: 1) официальный сайт продукта 2) HuggingFace/GitHub "
            "3) статья/пресс-релиз 4) 'отсутствует' если только t.me.\n"
            "• Разные источники с общим URL — объединяй в один FACT BLOCK.\n"
            "• Если в тексте нет конкретики — напиши 'NO CONCRETE DATA FOUND'. "
            "Не высасывай факты из пальца.\n\n"
            f"{reviews_section}\n\n"
            "=== ИСХОДНЫЙ ТЕКСТ ===\n"
            f"{news_feed}"
        ),
        expected_output=(
            "FACT BLOCK в формате === FACT BLOCK === ... === END FACT BLOCK ===. "
            "Нумерованные факты (минимум 3, если данные есть). Ничего лишнего."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Guardrail: function-based fact validation for Writer
# ---------------------------------------------------------------------------

def _make_fact_guardrail(research_task: Task):
    """Soft-warning guardrail: logs missing facts but NEVER blocks the post.

    Bot's job is to REWRITE news in its own words — not copy facts verbatim.
    Hard fact-gating was causing pipeline failures (3 retries → dead).
    Now we ALWAYS pass, but log warnings so humans can review if needed.

    Returns a function matching CrewAI's guardrail signature:
        (result: TaskOutput) -> Tuple[bool, Any]
    """
    def guardrail(result: TaskOutput):
        draft: str = result.raw or ""
        fact_block_raw: str = str(research_task.output) if research_task.output else ""

        if not fact_block_raw:
            return (True, draft)  # No FACT BLOCK to validate against — pass

        # ── Extract facts from FACTS section ──────────────────────────
        facts_section = re.search(
            r'FACTS:\n(.*?)(?:CAPABILITIES|LIMITATIONS|SOURCE_TONE|=== END)',
            fact_block_raw,
            re.DOTALL,
        )
        if not facts_section:
            return (True, draft)

        facts_text = facts_section.group(1)
        facts: list[str] = []
        for line in facts_text.strip().split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                fact = re.sub(r'^\d+\.\s*', '', line).strip()
                if fact and len(fact) > 8:
                    facts.append(fact)

        if not facts:
            return (True, draft)

        # ── Fuzzy check: each fact's key terms must appear in the draft ──
        missing: list[str] = []
        for fact in facts:
            key_terms = re.findall(
                r'[A-ZА-Я][a-zа-яё]{2,}|\d+[%×x]?|\b\w{6,}\b',
                fact,
            )
            if not key_terms:
                continue
            found = any(term.lower() in draft.lower() for term in key_terms)
            if not found:
                missing.append(fact)

        # ── Check URL presence ──────────────────────────────────────
        url_match = re.search(r'URL:\s*(.+?)(?:\n|$)', fact_block_raw)
        if url_match:
            fact_url = url_match.group(1).strip()
            if fact_url and fact_url != 'отсутствует':
                if 'http' not in draft.lower():
                    missing.append(f"ССЫЛКА НА ПРОДУКТ: {fact_url}")

        # ── SOFT WARNING — NEVER BLOCK ─────────────────────────────
        if missing:
            logger.warning(
                "Fact guardrail SOFT WARNING: %d facts may be missing/rewritten:\n%s\n\n"
                "Post is NOT blocked — rewriting in own words is EXPECTED behaviour.",
                len(missing),
                "\n".join(f"  - {f}" for f in missing),
            )

        # ALWAYS pass — bot rewrites, doesn't copy
        return (True, draft)

    return guardrail


def _write_task(agent: Agent, research_task: Task, post_type_name: str = "", post_type_desc: str = "") -> Task:
    return Task(
        description=(
            "Напиши пост для Telegram-канала на основе FACT BLOCK из контекста.\n\n"
            f"ТИП ПОСТА: {post_type_name}\n"
            f"{post_type_desc}\n\n"
            "═══ ФАКТЫ — ЭТО ТВОЙ ЕДИНСТВЕННЫЙ ИСТОЧНИК ═══\n"
            "• Ты получаешь FACT BLOCK от Researcher'а (в контексте).\n"
            "• ВСЕ факты из секции FACTS должны быть в посте. Каждый. Без исключений.\n"
            "• Ни одного факта из FACTS не теряй. Потерял факт — переписывай.\n"
            "• Ничего не добавляй сверх FACT BLOCK.\n"
            "• Если в FACT BLOCK написано 'NO CONCRETE DATA FOUND' — "
            "не выдумывай, напиши кратко что известно.\n"
            "• НЕЛЬЗЯ писать об отсутствующей информации: «подробностей пока нет», "
            "«деталей нет», «качество неизвестно», «ограничения не указаны». "
            "Если чего-то нет в FACT BLOCK — просто не упоминай это. Молчание лучше выдумки.\n"
            "• URL: Если в FACT BLOCK есть ссылка (поле URL, не 'отсутствует') — "
            "ОБЯЗАТЕЛЬНО включи её в пост. Вплети ссылку органично в текст: "
            "«запустили на krea.ai», «модель на HuggingFace», «релиз на GitHub». "
            "НИКОГДА не пиши отдельной строкой «Ссылка:», «Источник:», «Попробовать:». "
            "Ссылка должна быть частью предложения, а не отдельным блоком.\n\n"
            "• МЕДИА: Если в фиде есть пометка «⚠️ К этому посту прикреплено» — "
            "значит в оригинальном посте было изображение или видео. "
            "Можно упомянуть: «на видео», «на скриншоте», «в демо видно». "
            "НЕЛЬЗЯ писать: «все медиа загружены», «файлы прикреплены», «медиа сохранено» — "
            "это служебные фразы, в посте им не место.\n\n"
            "═══ ПРАВИЛА ═══\n"
            "КАТЕГОРИЧЕСКИЙ ЗАПРЕТ — ПЕРВОЕ ЛИЦО:\n"
            "• НИКАКИХ «я», «мне», «меня», «мой», «моё», «могу», «хочу».\n"
            "• НИКАКИХ «буду», «попробовал», «тестировал», «юзаю», «юзал».\n"
            "• НИКАКИХ «мне кажется», «я думаю», «я считаю».\n"
            "• НИКАКИХ «посмотрю», «потестирую», «покопаю», «расскажу».\n"
            "• Ты — не персонаж. Ты — lens, через которую читатель видит новость.\n"
            "• Писать БЕЗЛИЧНО: «выкатили модель», «интересная фича», «похоже на прорыв».\n\n"
            "ЗАПРЕЩЕНО (нарушение = брак):\n"
            "• Симулировать личный опыт: «я попробовал», «я тестировал».\n"
            "• Выдумывать планы: «буду тестировать», «надеюсь появится», «жду песочницу».\n"
            "• Писать про отсутствие: «нет API», «нет демо», «нет документации».\n"
            "• Бытовые аналогии: кухня, рация, автомобиль, тостер.\n"
            "  Сравнивай только с другими AI-инструментами.\n"
            "• Канцеляризмы: демонстрирует, представляет, является, позволяет, играет роль.\n"
            "• Служебные фразы про источники: «ссылаться на пост-источник», «источник: @канал».\n"
            "• Упоминать канал-источник в посте: «в канале X», «замечено в X», «X обратил внимание», «X пишет». Поле SOURCE в FACT BLOCK — только для справки, НИКОГДА не называй его в тексте поста.\n"
            "• Ссылки t.me/ в тексте поста: НЕЛЬЗЯ. Ссылки вида https://t.me/channel/123 — это технический маркер источника, они никогда не попадают в тело поста.\n"
            "• Добавлять советы, инструкции, вердикты, которых нет в FACT BLOCK. Нет факта → нет предложения.\n"
            "• Секции и заголовки «Вердикт», «Вывод», «Итог», «Мнение», «Оценка» в тексте поста — ЗАПРЕЩЕНЫ.\n"
            "• РЕДАКЦИОННЫЕ МНЕНИЯ, СКЕПТИЦИЗМ, ПРЕДСКАЗАНИЯ — БРАК:\n"
            "  Ни в конце, ни в середине поста нет места:\n"
            "  — «что же будет», «посмотрим как в деле», «остаётся только ждать»\n"
            "  — «поживём — увидим», «не врут ли», «верить ли», «а вдруг»\n"
            "  — «неизвестно взлетит ли», «конкуренты не дремлют»\n"
            "  — «пока непонятно», «время покажет», «рынок ответит»\n"
            "  — «есть уже похожее», «а получится ли у них»\n"
            "  Пост заканчивается ФАКТОМ. Никаких предсказаний и мнений.\n\n"
            "═══ AI-ШТАМПЫ — АВТОМАТИЧЕСКИЙ БРАК ═══\n"
            "Эти фразы НЕЛЬЗЯ использовать НИ В КАКОМ ВИДЕ:\n"
            "• «это не просто X, а Y» / «это уже не про X, а про Y»\n"
            "• «это меняет игру» / «меняет правила игры»\n"
            "• «открывает новые возможности» / «новые горизонты»\n"
            "• «ключевое:» / «самое интересное:» / «вот что зацепило»\n"
            "• «является свидетельством» / «свидетельствует о»\n"
            "• «давайте разберёмся» / «переходим к сути»\n"
            "• «эксперты считают» / «по мнению аналитиков»\n"
            "• «будущее выглядит» / «нас ждут» (пророчества)\n"
            "• «очередная модель» / «ещё одна нейросеть»\n"
            "• «без сомнения» / «нельзя не отметить» / «стоит отметить»\n"
            "• Любые вариации «является ключевым» / «играет роль»\n"
            "• «Попробовать:» / «Ссылка:» / «Источник:» — любые лейблы перед ссылкой\n"
            "• «Подробнее:» / «Читать:» / «Перейти:» — клише-призывы\n"
            "• «ссылаться на пост-источник» / «пост-источник» — служебная лексика\n\n"
            "═══ ГОЛОС ═══\n"
            "• Первая строка — факт, цифра, результат. Без предисловий.\n"
            "• Одна мысль, которая развивается. Живой русский с технической начинкой. "
            "Разговорные слова, сленг — ок.\n"
            "• КОРОТКАЯ НОВОСТЬ (FACT_COUNT ≤ 3): пост 300-600 знаков, 1-2 абзаца.\n"
            "• БОГАТАЯ НОВОСТЬ (FACT_COUNT ≥ 6): пост 1000-2000 знаков, 3-5 абзацев.\n"
            "  Если оригинальный пост богат техническими деталями — не сжимай их.\n"
            "  Бенчмарки + сравнения + параметры + цены = каждое в пост.\n"
            "• СРЕДНЯЯ НОВОСТЬ (FACT_COUNT 4-6): пост 600-1000 знаков.\n"
            "• ⛔ СЖИМАТЬ БОГАТУЮ НОВОСТЬ до 1 абзаца — БРАК.\n"
            "  Если в FACT BLOCK 8+ фактов, а пост вышел 400 знаков — переписывай.\n"
            "• ⛔ ЕСЛИ FACT_COUNT ≤ 2: пост НЕ БОЛЕЕ 450 знаков. ОДИН абзац.\n"
            "• БЕЗ ПЕРВОГО ЛИЦА. ВСЕГДА.\n"
            "ФОРМАТ ОТВЕТА — строго:\n\n"
            "ЧЕРНОВИК:\n"
            "[текст поста]\n\n"
            "ДЛИНА: пропорционально новости (короткая — короткий пост, богатая — длинный). Без хештегов."
        ),
        expected_output=(
            "ЧЕРНОВИК:\\n[текст поста]. Все факты из FACT BLOCK на месте, "
            "URL включён. "
            "Без первого лица. "
            "Пропорционально объёму новости."
        ),
        agent=agent,
        context=[research_task],
        guardrail=_make_fact_guardrail(research_task),
        guardrail_max_retries=3,
    )


def _edit_task(agent: Agent, write_task: Task, research_task: Task) -> Task:
    return Task(
        description=(
            "Твоя задача — факт-чекинг и минимальная редактура.\n\n"
            "У тебя есть:\n"
            "1. ЧЕРНОВИК от Writer'а (в контексте)\n"
            "2. FACT BLOCK от Researcher'а — эталон фактов (в контексте)\n\n"
            "ПРОВЕРЬ КАЖДЫЙ ПУНКТ (по порядку):\n"
            "□ ФАКТЫ: Каждый факт из секции FACTS (FACT BLOCK) есть в черновике? "
            "Если нет — ВЕРНИ факт в текст.\n"
            "□ ВЫДУМКИ: Есть ли в черновике то, чего НЕТ в FACT BLOCK? "
            "(метафоры, сценарии использования, приписанный опыт) — УДАЛИ.\n"
            "□ ЗАПРЕЩЁНКА: Проверь на отсутствие этих фраз:\n"
            "  — ПЕРВОЕ ЛИЦО: «я», «мне», «меня», «мой», «буду», «попробовал», «тестировал», «юзаю»\n"
            "  — «нет API», «нет демо», «нет документации»\n"
            "  — бытовые аналогии (кухня, еда, автомобиль)\n"
            "  — «надеюсь появится», «жду песочницу», «закрытая бета»\n"
            "  — «ссылаться на пост-источник», «источник: @канал»\n"
            "  Нашёл — УДАЛИ.\n"
            "□ НАЗВАНИЕ: Название модели/инструмента выделено в первой строке.\n"
            "□ ССЫЛКА: Если в FACT BLOCK поле URL ≠ 'отсутствует' — проверь что ссылка есть в посте. "
            "Если Writer её пропустил — ДОБАВЬ. Вплети ссылку в текст органично, "
            "как часть предложения. НИКОГДА не пиши 'Ссылка:', 'Источник:', 'Попробовать:' — "
            "ссылка должна быть частью живой речи.\n"
            "□ СКЕПТИЦИЗМ В КОНЦЕ: Проверь концовку — нет ли там мнений, предсказаний, "
            "сомнений? «Посмотрим», «время покажет», «неизвестно взлетит ли», "
            "«конкуренты не дремлют» — УДАЛИ. Концовка = последний факт.\n"
            "□ КЛИШЕ: «Это не просто X, а Y», «открывает новые горизонты», "
            "«меняет правила игры» — УДАЛИ.\n"
            "□ ГОЛОС: Живой язык? Короткие предложения? Не пресс-релиз? "
            "Не делай пост «гладким» за счёт живых интонаций.\n\n"
            "ЧЕГО НЕ ДЕЛАТЬ:\n"
            "• Не добавляй новые факты, сценарии, оценки.\n"
            "• Не переписывай голос автора — только правь факты и запрещёнку.\n"
            "• Не добавляй в пост служебную ссылку-источник («откуда взято», «источник: @канал», "
            "«ссылаться на пост-источник»). УПОМИНАНИЕ ИСТОЧНИКОВ В ТЕКСТЕ ПОСТА — ЗАПРЕЩЕНО.\n"
            "• Релевантные контентные ссылки (на модель, продукт, попробовать) — "
            "ОБЯЗАТЕЛЬНО ДОБАВЬ, если Writer их упустил. Читатель должен иметь возможность "
            "перейти и попробовать то, о чём пост.\n"
            "ФОРМАТ ОТВЕТА — строго:\n\n"
            "ФИНАЛЬНЫЙ ПОСТ:\n"
            "[текст поста]"
        ),
        expected_output=(
            "ФИНАЛЬНЫЙ ПОСТ:\\n[текст]. Все факты из FACT BLOCK на месте. "

        ),
        agent=agent,
        context=[write_task, research_task],
    )


def _rhythm_task(agent: Agent, edit_task: Task, recent_posts: str) -> Task:
    return Task(
        description=(
            "Получи финальный пост от Editor'а и сравни с последними постами канала.\n\n"
            "ПРОВЕРЬ:\n"
            "• Не повторяется ли тип хука с последними 3 постами?\n"
            "• Не повторяется ли тональность (3 подряд скептических/восторженных)?\n"
            "• Нет ли повторяющихся фраз или конструкций?\n"
            "• Не совпадает ли ритмический рисунок?\n\n"
            "ДЕЙСТВИЕ:\n"
            "• Если пост уникален по ритму — верни без изменений.\n"
            "• Если есть повторение — минимальная правка: другой хук или другой финал.\n"
            "• Голос автора не менять. Факты не добавлять.\n\n"
            "=== ПОСЛЕДНИЕ ПОСТЫ КАНАЛА ===\n"
            f"{recent_posts if recent_posts.strip() else '(нет данных)'}\n"
            "=== КОНЕЦ ПОСЛЕДНИХ ПОСТОВ ===\n\n"
            "ФОРМАТ ОТВЕТА — строго:\n\n"
            "ФИНАЛЬНЫЙ ПОСТ:\n"
            "[текст поста]"
        ),
        expected_output=(
            "ФИНАЛЬНЫЙ ПОСТ:\\n[текст]. Ритм проверен. "
            "Без изменений ИЛИ с минимальной правкой."
        ),
        agent=agent,
        context=[edit_task],
    )


def _format_task(agent: Agent, rhythm_task: Task) -> Task:
    return Task(
        description=(
            "Оформи пост визуально по правилам из SKILL.md.\n\n"
            "ЧТО СДЕЛАТЬ (по порядку):\n"
            "1. Если есть заголовок/первая фраза — оберни в <b>текст</b>.\n"
            "   ВАЖНО: В конце жирного заголовка НЕ ставить точку.\n"
            "   После жирного заголовка — ОБЯЗАТЕЛЬНО пустая строка перед телом поста.\n"
            "2. Добавь 1-3 органичных эмодзи по тексту (не буллеты).\n"
            "3. Проверь пустые строки между абзацами — должна быть пустая строка.\n"
            "4. Если есть промпт длиннее 3 строк — оберни в:\n"
            "   <blockquote expandable>промпт</blockquote>\n"
            "5. Короткие цитаты (1-3 строки) — в <blockquote>цитата</blockquote>.\n\n"
            "НЕЛЬЗЯ:\n"
            "• Менять содержание, факты, голос автора.\n"
            "• Добавлять хештеги.\n"
            "• Ставить <b> на обычные предложения — только заголовок.\n"
            "• Ставить эмодзи в начале каждого абзаца.\n"
            "• Удалять или перефразировать факты.\n\n"
            "ССЫЛКИ — ОБЯЗАТЕЛЬНО:\n"
            "\u2022 Markdown-ссылка [текст](url) \u2192 конвертируй в HTML: <a href=\"url\">текст</a>\n"
            "\u2022 Голая ссылка рядом со словом-якорем в скобках \u2014 объединяй в <a href=\"url\">якорь</a>\n"
            "\u2022 Уже готовые <a href=\"...\">..</a> \u2014 не трогай.\n\n"
            "ФОРМАТ ОТВЕТА — строго:\n\n"
            "ФИНАЛЬНЫЙ ПОСТ:\n"
            "[оформленный текст поста]"
        ),
        expected_output=(
            "ФИНАЛЬНЫЙ ПОСТ:\\n[текст]. Заголовок в <b>, "
            "1-3 эмодзи, пустые строки, корректные blockquote."
        ),
        agent=agent,
        context=[rhythm_task],
    )


def _visual_task(agent: Agent, final_task: Task) -> Task:
    return Task(
        description=(
            "Создай ОДИН английский промпт для генерации обложки поста.\n\n"
            "Требования:\n"
            "• Английский язык\n"
            "• Конкретный: стиль, атмосфера, объекты, освещение, цветовая палитра\n"
            "• 40-80 слов\n"
            "• Готов для Midjourney / DALL·E / Stable Diffusion\n\n"
            "ФОРМАТ ОТВЕТА — строго:\n\n"
            "ПРОМПТ ДЛЯ ИЗОБРАЖЕНИЯ:\n"
            "[english prompt]"
        ),
        expected_output=(
            "ПРОМПТ ДЛЯ ИЗОБРАЖЕНИЯ:\\n[english prompt]. 40-80 слов."
        ),
        agent=agent,
        context=[final_task],
    )


# ---------------------------------------------------------------------------
# Output parsers (minimal — no regex AI-pattern cleanup)
# ---------------------------------------------------------------------------

def _parse_variants(raw: str) -> list[str]:
    """
    Extract the final post text.
    Primary format: 'ФИНАЛЬНЫЙ ПОСТ:\\n...'
    Fallback: 'ЧЕРНОВИК:\\n...'
    Last resort: return raw text as-is.

    Handles NESTED markers: when agents stack (Editor→Rhythm→Formatter),
    each may prepend its own marker. We take the LAST marker occurrence
    (the innermost text), which is the actual post content.
    """
    for marker in (r"ФИНАЛЬНЫЙ\s+ПОСТ\s*:\s*\n?", r"ЧЕРНОВИК\s*:\s*\n?"):
        # Find ALL occurrences, take the LAST one (innermost content)
        matches = list(re.finditer(marker, raw, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            text = raw[last_match.end():].strip()
            # Strip anything after ПРОМПТ ДЛЯ ИЗОБРАЖЕНИЯ
            text = re.sub(
                r"\n{0,2}ПРОМПТ\s+ДЛЯ\s+ИЗОБРАЖЕНИЯ.*$",
                "",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            if text:
                return [_clean_post_text(text)]

    logger.warning("_parse_variants: no format marker found; using raw output")
    return [_clean_post_text(raw.strip())]


def _parse_image_prompt(raw: str) -> str | None:
    match = re.search(
        r"ПРОМПТ\s+ДЛЯ\s+ИЗОБРАЖЕНИЯ\s*:\s*\n?(.*)",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

# Minimal cleanup — only structural issues, not "AI patterns"
# If the model generates bad content, fix the prompt, not the output.
_SERVICE_PHRASES_CLEANUP = [
    r"(?i)голос\s+(?:натали|автора)\s+(?:сохран[её]н|проверен)[,.]?\s*",
    r"(?i)ритм\s+(?:канала\s+)?проверен[,.]?\s*",
    r"(?i)===?\s*FACT\s+BLOCK\s*===?[\s\S]*?===?\s*END\s+FACT\s+BLOCK\s*===?",
    r"(?i)===?\s*ИСХОДНЫЙ\s+ТЕКСТ\s*===?[\s\S]*?(?=ЧЕРНОВИК|ФИНАЛЬНЫЙ|$)",
    r"(?i)in\s+(?:this\s+)?(?:task|post)[,.]?\s*(?:you['´`]?ll|i['´`]?ll|we['´`]?ll)",
    # Служебные маркеры, просочившиеся в текст
    # Агрессивная чистка: удаляем строки с маркерами ГДЕ УГОДНО в тексте,
    # не только в начале строки. LLM часто вставляет их в середину поста.
    r"(?im)[^\n]*ФИНАЛЬНЫЙ\s+ПОСТ[^\n]*\n?",
    r"(?im)^ФИНАЛЬНОСТЬ\s*:[^\n]*\n?",
    r"(?im)^ИТОГ\s*:[^\n]*\n?",
    r"(?im)[^\n]*ЧЕРНОВИК\s*:[^\n]*\n?",
    r"(?i)без\s+изменений\s*(?:или\s+с\s+минимальной\s+правкой)?[.,]?\s*$",
    r"(?i)^\s*ДЛИНА\s*:\s*\d+[-–]\d+\s+символов[.,]?\s*",
    # Research-whining: strip lines claiming info is absent
    r"(?i)[^\n]*подробностей\s+пока\s+нет[^\n]*",
    r"(?i)[^\n]*деталей\s+пока\s+нет[^\n]*",
    r"(?i)[^\n]*(?:качестве|ограничениях)\s+пока\s+нет[^\n]*",
    # AI-stamp link labels
    r"(?im)^\s*(?:Попробовать|Ссылка|Источник|Подробнее|Читать|Перейти)\s*:.*$",
    # AI-stamp phrases — remove whole lines containing them
    r"(?i)[^\n]*ссылаться\s+на\s+пост[-\s]?источник[^\n]*",
    r"(?i)[^\n]*пост[-\s]?источник[^\n]*",
    r"(?i)[^\n]*источник:\s*@[^\n]*",
    r"(?i)[^\n]*https?://t\.me/[A-Za-z0-9_]+/[0-9]+[^\n]*",
    r"(?im)^[\*_]*(?:Вердикт|Вывод|Итог|Мнение|Оценка)[\*_]*\s*:[^\n]*\n?",
    r"(?im)^[\*_]*(?:Вердикт|Вывод|Итог|Мнение|Оценка)[\*_]*\s*$\n?",
    r"(?i)[^\n]*откуда\s+взято[^\n]*",
    # First-person patterns — REMOVE lines containing them.
    # These are the same patterns _validate_final_post currently only logs.
    # The LLM ignores prompt-level bans, so output-level cleanup is necessary.
    r"(?i)[^\n]*\bя\s+(?:попробовал[а]?|тестировал[а]?|юзал[а]?|юзаю|думаю|считаю|посмотрю|потестирую|покопаю|расскажу|сделал[а]?|проверил[а]?|нашел[а]?|наш[её]л|заметил[а]?)\b[^\n]*",
    r"(?i)[^\n]*\b(?:буду|собираюсь)\s+(?:юзать|пробовать|тестировать|тестить|смотреть|копать)\b[^\n]*",
    r"(?i)[^\n]*\bмне\s+кажется\b[^\n]*",
    r"(?i)[^\n]*\bмой\s+(?:опыт|тест|прогон|запуск)\b[^\n]*",
    # AI-stamp phrases — additional patterns that slip through prompt bans
    r"(?i)[^\n]*\b(?:давайте|давай)\s+(?:разбер[её]мся|посмотрим|пойм[её]м|обсудим)\b[^\n]*",
    r"(?i)[^\n]*\bвот\s+что\s+(?:нужно|важно|стоит)\s+(?:знать|понимать|учитывать)\b[^\n]*",
    r"(?i)[^\n]*\b(?:без\s+сомнения|нельзя\s+не\s+отметить|стоит\s+отметить|следует\s+отметить)\b[^\n]*",
    r"(?i)[^\n]*\b(?:эксперты\s+(?:считают|говорят|отмечают)|по\s+мнению\s+(?:экспертов|аналитиков))\b[^\n]*",
    r"(?i)[^\n]*\b(?:открывает|открывают)\s+новые\s+(?:возможности|горизонты|перспективы)\b[^\n]*",
    r"(?i)[^\n]*\bменяет\s+правила\s+игры\b[^\n]*",
    r"(?i)[^\n]*\b(?:очередная|ещ[её]\s+одна)\s+(?:модель|нейросеть|инструмент|сервис|релиз)\b[^\n]*",
    r"(?i)[^\n]*\b(?:играет|является)\s+(?:ключевую|важную|значимую|решающую)\s+роль\b[^\n]*",
    r"(?i)[^\n]*\b(?:свидетельствует|является\s+свидетельством)\s+о\b[^\n]*",
    r"(?i)[^\n]*\bэто\s+(?:уже\s+)?не\s+(?:просто|про)\s+\w+[,.]?\s*(?:а|это)\b[^\n]*",
    r"(?i)[^\n]*\b(?:будущее|нас)\s+(?:выглядит|жд[её]т)\s+(?:многообещающ|светл|захватывающ|интересн)\b[^\n]*",
    # Skeptical editorial endings
    r"(?i)[^\n]*\b(?:поживём|поживем)\s*[—–-]\s*увидим\b[^\n]*",
    r"(?i)[^\n]*\bвремя\s+покажет\b[^\n]*",
    r"(?i)[^\n]*\bостаётся\s+только\s+ждать\b[^\n]*",
    r"(?i)[^\n]*\bнеизвестно\s+(?:получится|взлетит|выстрелит|удастся)\b[^\n]*",
    r"(?i)[^\n]*\b(?:рынок|конкуренты)\s+(?:ответит|не\s+дремлют|посмотрим)\b[^\n]*",
    r"(?i)[^\n]*\bпока\s+(?:непонятно|неясно)\b[^\n]*",
    r"(?i)[^\n]*\bа\s+получится\s+ли\b[^\n]*",
    r"(?i)[^\n]*\bпосмотрим\s+(?:как|что)\b[^\n]*",
    # Service messages about media that leak into post text
    r"(?im)^[^\n]*(?:медиа|видео|фото)\s+(?:загружен|прикреплен|сохранен)[^\n]*$",
    r"(?im)^[^\n]*все\s+(?:мультимедиа|медиа\s+файлы)\s+загружен[^\n]*$",
    r"(?im)^[^\n]*изображени[еяй]\s+(?:загружен|прикреплен|сохранен)[^\n]*$",
    # "Это не просто X, а Y" construction — AI cliché
    r"(?i)[^\n]*это\s+не\s+просто\s+\w+[,.]?\s*(?:это|а)\s+[^\n]*",
    r"(?i)[^\n]*это\s+уже\s+не\s+(?:про|о)\s+\w+[,.]?\s*(?:это|а)\s+[^\n]*",
]
def _clean_post_text(text: str) -> str:
    """Remove service artifacts, no AI-pattern censorship."""
    for pattern in _SERVICE_PHRASES_CLEANUP:
        text = re.sub(pattern, "", text)
    # Clean up whitespace artifacts
    # Convert Markdown links [text](url) → <a href="url">text</a>
    text = re.sub(r'\[([^\]]+)\]\s*\((https?://[^)]+)\)', r'<a href="\2">\1</a>', text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{2,}$", "", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    # Fix header formatting: strip trailing punctuation from <b>...</b> blocks
    # "текст<b>Заголовок.</b>" → "текст<b>Заголовок</b>"
    text = re.sub(
        r'<b>([^<]+?)([.,:;!?…]+)</b>',
        r'<b>\1</b>',
        text,
    )
    # Fix: <b>Header</b> then regular text on SAME line → split with blank line
    # "текст<b>Заголовок</b> Продолжение" → "текст<b>Заголовок</b>\n\nПродолжение"
    text = re.sub(
        r'</b>\s+(?!\n)([^\n])',
        r'</b>\n\n\1',
        text,
    )
    # Ensure empty line after </b> at end of line: add \n\n if only single \n
    # "текст</b>\nПродолжение" → "текст</b>\n\nПродолжение"
    text = re.sub(
        r'</b>\n(?!\n)([^\n])',
        r'</b>\n\n\1',
        text,
    )
    # Strip trailing punctuation from bold text at very end of string
    # "<b>Заголовок.</b>" → "<b>Заголовок</b>"
    text = re.sub(
        r'<b>([^<]+?)([.,:;!?…]+)</b>$',
        r'<b>\1</b>',
        text,
    )
    # Fix emoji-before-period: "текст 😊." → "текст. 😊"
    # Handles both direct (😊.) and space-separated (😊 .) cases
    # Also handles multiple emoji: "🔥😊." → ".🔥😊"
    text = re.sub(
        r'([🌀-🧿☀-➿⭐❤😀-🙏🚀-🛿'
        r'✂-➰©®™〰〽㊗㊙'
        r'🤀-🧿🨀-🩯🩰-🫿'
        r'🀄-🃏‍️️]+)\s*\.(\s|$)',
        r'.\1\2',
        text,
    )
    # Also fix: emoji before period with no space before next word
    # "текст😊.следующее" → "текст. 😊следующее" → then paragraph spacing fixes it
    text = re.sub(
        r'([🌀-🧿☀-➿⭐❤😀-🙏🚀-🛿'
        r'✂-➰©®™〰〽㊗㊙'
        r'🤀-🧿🨀-🩯🩰-🫿'
        r'🀄-🃏‍️️]+)\.(\S)',
        r'.\1 \2',
        text,
    )
    # Enforce paragraph spacing: single \n between non-empty lines → \n\n
    # Skip list items (starting with -, •, *, digit.) to avoid breaking formatting
    lines = text.split("\n")
    result: list[str] = []
    for i, line in enumerate(lines):
        result.append(line)
        if i < len(lines) - 1:
            this_stripped = line.strip()
            next_stripped = lines[i + 1].strip()
            # Both lines non-empty, neither is a list item
            is_list = lambda s: bool(re.match(r"^[-•*]\s|^\d+[.)]\s", s))
            if (this_stripped and next_stripped
                    and not is_list(this_stripped)
                    and not is_list(next_stripped)):
                result.append("")  # add extra blank line
    text = "\n".join(result)
    # Final cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(".,:;\n ").strip()


def _strip_tg_links(text: str) -> str:
    """Remove Telegram channel links and @mentions from post text.
    
    Keeps t.me links that point to SPECIFIC POSTS (contain a message ID),
    as these are product/content links, not channel references.
    Only strips bare channel links (t.me/channel without /message_id).
    """
    # Remove bare channel links: t.me/channel (no message ID after)
    text = re.sub(r'https?://t\.me/[A-Za-z0-9_]+/?\s', ' ', text)
    text = re.sub(r'https?://t\.me/[A-Za-z0-9_]+/?$', '', text)
    # Remove @username mentions
    text = re.sub(r'(?<!\S)@[A-Za-z][A-Za-z0-9_]{3,}(?!\S)', '', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _validate_final_post(text: str) -> list[str]:
    """
    Lightweight validation. Returns list of violations.
    Empty list = clean. Only checks for hard no-gos.
    """
    violations = []
    if not text:
        return violations

    text_lower = text.lower()

    # Check for research-whining (invented absence of info)
    for phrase in ["нет api", "нет demo", "закрытая бета",
                   "подробностей пока нет", "деталей пока нет",
                   "подробности пока", "деталей нет",
                   "качестве пока нет", "ограничениях пока нет",
                   "информации пока нет", "данных пока нет"]:
        if phrase in text_lower:
            violations.append(f"RESEARCH-WHINING: '{phrase}'")

    # Check for fake personal experience
    for phrase in ["буду тестировать", "жду документацию", "надеюсь появится"]:
        if phrase in text_lower:
            violations.append(f"FAKE PLAN: '{phrase}'")

    # Check for first-person (CRITICAL)
    for phrase in ["я попробовал", "я тестировал", "я юзал", "я юзаю",
                   "мне кажется", "я думаю", "я считаю", "я посмотрю",
                   "я потестирую", "я покопаю", "я расскажу",
                   "буду юзать", "буду пробовать", "буду тестить",
                   "попробовала", "тестировала", "попробовал"]:
        if phrase in text_lower:
            violations.append(f"FIRST-PERSON: '{phrase}'")

    # Check for AI-stamp link labels
    for phrase in ["попробовать:", "источник:", "подробнее:", "читать:", "перейти:",
                   "ссылка:", "source:", "try:"]:
        if phrase in text_lower:
            violations.append(f"AI-STAMP: '{phrase}'")

    # Check for source references
    for phrase in ["пост-источник", "ссылаться на", "источник: @",
                   "откуда взято"]:
        if phrase in text_lower:
            violations.append(f"SOURCE-REF: '{phrase}'")

    return violations


# ---------------------------------------------------------------------------
# Synchronous crew runner
# ---------------------------------------------------------------------------

def _run_crew_sync(
    news_items: list[dict],
    recent_posts: str = "",
    user_reviews: str = "",
) -> PipelineResult:
    if not news_items:
        raise ValueError("news_items is empty — nothing to process")

    def _fmt_item(i: int, item: dict) -> str:
        date_str = str(item.get("date", "неизвестно"))[:25]
        has_media = item.get("media_path") or item.get("has_media")
        media_note = ""
        if has_media:
            media_type_hint = item.get("media_type", "")
            if media_type_hint == "video":
                media_note = "\n⚠️ К этому посту прикреплено ВИДЕО. ОБЯЗАТЕЛЬНО опиши его содержание в посте."
            elif media_type_hint == "photo":
                media_note = "\n⚠️ К этому посту прикреплено ИЗОБРАЖЕНИЕ. Опиши, что на нём."
            else:
                media_note = "\n⚠️ К этому посту прикреплено медиа (фото/видео). Упомяни это в посте."
        raw_url: str = item.get("url", "")
        if raw_url:
            url_line = f"URL источника: {raw_url}"
        else:
            url_line = "URL источника: отсутствует"
        return (
            f"[{i+1}] {item.get('source', '')}\n"
            f"Дата: {date_str}\n"
            f"{url_line}{media_note}\n\n"
            f"{item.get('text', '')[:1500]}"
        )

    news_feed = "\n\n".join(
        _fmt_item(i, item) for i, item in enumerate(news_items[:30])
    )

    # ── Temporal Context Injection ────────────────────────────────────────
    from datetime import datetime as _dt
    _today = _dt.now().strftime("%d %B %Y")
    news_feed = (
        f"══ ДАТА: {_today} ══\n"
        f"Оценивай актуальность материалов относительно этой даты.\n\n"
        f"{news_feed}"
    )

    researcher      = _make_researcher(_llm(temperature=0.3))  # low temp: precise extraction
    writer          = _make_writer(_llm(temperature=0.5))      # balanced: facts + voice
    editor          = _make_editor(_llm(temperature=0.3))       # low temp: strict fact-check
    rhythm_checker  = _make_rhythm_checker(_llm(temperature=0.7))
    formatter       = _make_formatter(_llm(temperature=0.5))
    visual_designer = _make_visual_designer(_llm(temperature=0.8)) if ENABLE_VISUAL_PROMPT else None

    r_task   = _research_task(researcher, news_feed, user_reviews)
    pt_name, pt_desc = _select_post_type()
    logger.info("WriterCrew: selected post type: %s", pt_name)
    w_task   = _write_task(writer, r_task, post_type_name=pt_name, post_type_desc=pt_desc)
    e_task   = _edit_task(editor, w_task, r_task)
    rh_task  = _rhythm_task(rhythm_checker, e_task, recent_posts)
    fmt_task = _format_task(formatter, rh_task)
    v_task   = _visual_task(visual_designer, fmt_task) if visual_designer else None  # type: ignore[arg-type]

    agents = [researcher, writer, editor, rhythm_checker, formatter]
    tasks  = [r_task, w_task, e_task, rh_task, fmt_task]
    if visual_designer and v_task:
        agents.append(visual_designer)
        tasks.append(v_task)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )
    crew.kickoff()

    researcher_summary = str(r_task.output) if r_task.output else ""
    format_raw         = str(fmt_task.output) if fmt_task.output else ""
    visual_raw         = str(v_task.output) if (v_task and v_task.output) else ""

    variants     = [_strip_tg_links(v) for v in _parse_variants(format_raw)]
    image_prompt = _parse_image_prompt(visual_raw) if visual_raw else None

    # Lightweight post-generation validation
    if variants:
        violations = _validate_final_post(variants[0])
        if violations:
            logger.warning(
                "WriterCrew validation FAILED: violations=%s | post_preview=%r",
                violations,
                variants[0][:200],
            )
        else:
            logger.info("WriterCrew validation passed")

    # Post size proportionality check
    if variants:
        post_len = len(variants[0])
        # Estimate news volume from FACT BLOCK fact count
        fact_count = 0
        if researcher_summary:
            facts_section = re.search(
                r"FACTS:\n(.*?)(?:CAPABILITIES|LIMITATIONS|SOURCE_TONE|=== END)",
                researcher_summary, re.DOTALL,
            )
            if facts_section:
                fact_count = len(re.findall(r"^\d+\.", facts_section.group(1), re.MULTILINE))
        if fact_count <= 2 and post_len > 600:
            logger.warning(
                "WriterCrew SIZE MISMATCH: %d facts → post is %d chars (should be ≤400 for ≤2 facts). "
                "Post may be bloated.",
                fact_count, post_len,
            )
        elif fact_count <= 4 and post_len > 1200:
            logger.warning(
                "WriterCrew SIZE MISMATCH: %d facts → post is %d chars (max recommended ~900). "
                "Post may be overlong.",
                fact_count, post_len,
            )

    logger.info(
        "WriterCrew finished: post_len=%d chars, image_prompt=%s",
        len(variants[0]) if variants else 0,
        "yes" if image_prompt else "no",
    )
    return PipelineResult(
        variants=variants,
        image_prompt=image_prompt,
        researcher_summary=researcher_summary,
    )


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def run_pipeline(
    news_items: list[dict],
    recent_posts: str = "",
    user_reviews: str = "",
    style_examples: str = "",  # deprecated, kept for backwards compatibility
) -> PipelineResult:
    """
    Run the WriterCrew pipeline in a thread executor.

    Args:
        news_items:    filtered post list from fetch_all_sources
        recent_posts:  last 5-10 published posts from the channel (plain text),
                       used by RhythmChecker to avoid repeating patterns.
                       Format: each post separated by '\\n---\\n'
        user_reviews:  community/user feedback on the topic (optional)

    Returns PipelineResult with one final post text + optional image prompt.
    """
    logger.info("WriterCrew: starting for %d news items", len(news_items))
    loop = asyncio.get_event_loop()
    result: PipelineResult = await loop.run_in_executor(
        None,
        partial(_run_crew_sync, news_items, recent_posts, user_reviews),
    )
    return result
