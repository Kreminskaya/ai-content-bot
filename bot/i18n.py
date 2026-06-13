"""
Bilingual interface (RU / EN) for the whole bot — like a language picker in a game.

One switch controls everything the user sees: command menu, buttons, status
messages, the digest layout, and the language of generated content.

The choice is made at runtime via the /language command (handlers.py) and is
remembered across restarts in a small JSON state file. The initial default
comes from the LANGUAGE (or legacy CONTENT_LANGUAGE) env var, falling back to ru.

Usage:
    from bot.i18n import t, get_language, set_language, content_language_name
    await message.answer(t("running_pipeline"))
    await message.answer(t("draft_label", id=42))
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Runtime state file (the remembered language). Not committed — it's per-install state.
_STATE_FILE = Path(__file__).resolve().parent.parent / "runtime_state.json"

# Initial default: LANGUAGE wins, then legacy CONTENT_LANGUAGE, then Russian.
_ENV_DEFAULT = os.getenv("LANGUAGE") or os.getenv("CONTENT_LANGUAGE") or "ru"


def _normalize(value: str | None) -> str:
    """Map any spelling ('en', 'EN', 'English', 'английский') to 'en' or 'ru'."""
    v = (value or "").strip().lower()
    if v.startswith("en") or "english" in v or "англ" in v:
        return "en"
    return "ru"


def get_language() -> str:
    """Return the current UI/content language: 'ru' or 'en'."""
    try:
        if _STATE_FILE.exists():
            data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            if data.get("language"):
                return _normalize(data["language"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("i18n: could not read state file: %s", exc)
    return _normalize(_ENV_DEFAULT)


def set_language(value: str) -> str:
    """Persist the chosen language. Returns the normalized value ('ru'/'en')."""
    lang = _normalize(value)
    try:
        data: dict = {}
        if _STATE_FILE.exists():
            data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        data["language"] = lang
        _STATE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("i18n: could not write state file: %s", exc)
    return lang


def content_language_name() -> str:
    """Full language name to inject into LLM prompts (digest/post generation)."""
    return "English" if get_language() == "en" else "Russian"


# ---------------------------------------------------------------------------
# All user-facing strings: key -> (Russian, English)
# Placeholders use {name} and are filled via t("key", name=...).
# ---------------------------------------------------------------------------

STRINGS: dict[str, tuple[str, str]] = {
    # ── Language picker ──────────────────────────────────────────────────────
    "choose_language": (
        "🌐 Выбери язык интерфейса:",
        "🌐 Choose the interface language:",
    ),
    "language_set": (
        "✅ Язык переключён на русский. Весь интерфейс и контент теперь на русском.",
        "✅ Language switched to English. The whole interface and content are now in English.",
    ),
    "btn_lang_ru": ("🇷🇺 Русский", "🇷🇺 Русский"),
    "btn_lang_en": ("🇬🇧 English", "🇬🇧 English"),

    # ── Commands / menu descriptions ─────────────────────────────────────────
    "cmd_start_desc":    ("Главное меню", "Main menu"),
    "cmd_run_desc":      ("Запустить дайджест вручную", "Run the digest manually"),
    "cmd_post_desc":     ("✍️ Написать пост по своей новости", "✍️ Write a post from your news"),
    "cmd_status_desc":   ("Последние посты и дайджесты", "Recent posts and digests"),
    "cmd_cancel_desc":   ("Отменить текущее действие", "Cancel the current action"),
    "cmd_language_desc": ("🌐 Сменить язык (RU/EN)", "🌐 Change language (RU/EN)"),

    # ── /start ───────────────────────────────────────────────────────────────
    "start_message": (
        "👋 Привет! Я твой AI-редактор для Telegram-канала.\n\n"
        "<b>Команды:</b>\n"
        "/run — запустить дайджест-пайплайн вручную\n"
        "/post — ✍️ написать пост по своей новости\n"
        "/status — последние посты и дайджесты\n"
        "/language — 🌐 сменить язык (RU/EN)\n\n"
        "Когда придёт дайджест — нажми на нужную тему, бот напишет пост.\n"
        "Потом: <b>✅ Постить</b> — публикация, "
        "<b>✏️ Редактировать</b> — правка, "
        "<b>🎨 Промпт</b> — картинка, "
        "<b>❌ Отмена</b> — пропустить.",
        "👋 Hi! I'm your AI editor for the Telegram channel.\n\n"
        "<b>Commands:</b>\n"
        "/run — run the digest pipeline manually\n"
        "/post — ✍️ write a post from your own news\n"
        "/status — recent posts and digests\n"
        "/language — 🌐 change language (RU/EN)\n\n"
        "When a digest arrives — tap a topic and the bot will write a post.\n"
        "Then: <b>✅ Publish</b> — publish, "
        "<b>✏️ Edit</b> — edit, "
        "<b>🎨 Image prompt</b> — image, "
        "<b>❌ Cancel</b> — skip.",
    ),

    # ── /post (manual mode) ──────────────────────────────────────────────────
    "manual_intro": (
        "✍️ <b>Ручной режим — новый пост</b>\n\n"
        "Пришли мне текст новости, ссылку или перешли сообщение, "
        "по которому нужно сделать пост.\n\n"
        "<i>/cancel — отмена</i>",
        "✍️ <b>Manual mode — new post</b>\n\n"
        "Send me the news text, a URL, or forward a message "
        "to create a post from.\n\n"
        "<i>/cancel — cancel</i>",
    ),
    "manual_no_text": (
        "Не получил текст. Пришли новость текстом, ссылкой или перешли сообщение.\n"
        "Попробуй снова: /post",
        "No text received. Send the news as text, a URL, or a forwarded message.\n"
        "Try again: /post",
    ),
    "manual_cancelled": ("✍️ Ручной режим отменён.", "✍️ Manual mode cancelled."),

    # ── Edit flow ────────────────────────────────────────────────────────────
    "edit_mode": (
        "✏️ <b>Режим редактирования</b>\n\n"
        "Ниже — текст поста как он выглядит сейчас.\n"
        "Скопируй, отредактируй и пришли обратно — я опубликую в канал с сохранением "
        "всего форматирования и премиум-эмодзи.\n\n"
        "/cancel — отмена",
        "✏️ <b>Edit mode</b>\n\n"
        "Below is the current post text.\n"
        "Copy it, edit it, and send it back — I'll publish it to the channel keeping "
        "all formatting and premium emoji.\n\n"
        "/cancel — cancel",
    ),
    "edit_cancelled": ("Редактирование отменено.", "Editing cancelled."),
    "text_empty": (
        "Текст не должен быть пустым. Попробуй ещё раз или /cancel.",
        "Text must not be empty. Try again or /cancel.",
    ),
    "text_saved": (
        "<b>✅ Текст сохранён.</b>\nНажми «Постить», чтобы опубликовать в канал.",
        "<b>✅ Text saved.</b>\nClick «Publish» to post to the channel.",
    ),
    "text_is_empty": ("Текст пустой.", "Text is empty."),
    "edit_in_progress": (
        "⚠️ Пост сейчас редактируется. Нажми «Постить» в сообщении "
        "с подтверждением редактирования.",
        "⚠️ Post is being edited. Click «Publish» in the edit confirmation message.",
    ),
    "use_new_confirm": (
        "⚠️ Используй кнопку «Постить» в НОВОМ сообщении с подтверждением.",
        "⚠️ Use the «Publish» button in the NEW confirmation message.",
    ),

    # ── Pipeline / writing ───────────────────────────────────────────────────
    "running_pipeline": ("⚙️ Запускаю дайджест-пайплайн…", "⚙️ Running the digest pipeline…"),
    "writing_post": ("⏳ <b>Пишу пост…</b>", "⏳ <b>Writing the post…</b>"),
    "writing_post_toast": ("⏳ Пишу пост…", "⏳ Writing the post…"),
    "writing_on_topic": ("⏳ <b>Пишу пост по теме:</b>", "⏳ <b>Writing post on topic:</b>"),
    "no_variants": (
        "⚠️ Не удалось сгенерировать пост. Проверь логи.",
        "⚠️ No post variants returned. Check the logs.",
    ),
    "media_received": (
        "<i>Получено медиа из пересланного сообщения: {n}.</i>",
        "<i>Received {n} media file(s) from the forwarded message.</i>",
    ),
    "run_error": ("❌ Ошибка:\n<code>{err}</code>", "❌ Error:\n<code>{err}</code>"),
    "writing_error": (
        "❌ Ошибка при написании поста:\n<code>{err}</code>",
        "❌ Error while writing the post:\n<code>{err}</code>",
    ),
    "llm_limit": (
        "❌ Лимит LLM-провайдера исчерпан.\n"
        "Нужно переключить модель/ключ или дождаться сброса лимита.",
        "❌ LLM provider limit reached.\n"
        "Switch the model/key or wait for the limit to reset.",
    ),
    "llm_limit_fallback": (
        "❌ Лимит LLM-провайдера исчерпан.\n"
        "Провайдер сейчас недоступен для генерации. Переключаемся на резервную модель.",
        "❌ LLM provider limit reached.\n"
        "The provider is temporarily unavailable. Switching to a fallback model.",
    ),

    # ── Buttons ──────────────────────────────────────────────────────────────
    "btn_publish":    ("✅ Постить", "✅ Publish"),
    "btn_edit":       ("✏️ Редактировать", "✏️ Edit"),
    "btn_cancel":     ("❌ Отмена", "❌ Cancel"),
    "btn_reject_all": ("❌ Отклонить всё", "❌ Reject all"),
    "btn_img_prompt": ("🎨 Промпт для картинки", "🎨 Image prompt"),

    # ── Digest layout ────────────────────────────────────────────────────────
    "digest_header":     ("📰 <b>Дайджест за {date}</b>", "📰 <b>Digest for {date}</b>"),
    "section_hard":      ("<b>── 🧠 Хард-Новости ──</b>", "<b>── 🧠 Breaking News ──</b>"),
    "section_production":("<b>── 🎬 Кейсы Продакшена ──</b>", "<b>── 🎬 Production Cases ──</b>"),
    "section_useful":    ("<b>── 🛠 Полезности ──</b>", "<b>── 🛠 Useful Tools ──</b>"),
    "digest_footer": (
        "👆 Нажми на тему, чтобы написать пост ({n} тем)",
        "👆 Tap a topic to write a post ({n} topics)",
    ),
    "topic_fallback": ("Тема {idx}", "Topic {idx}"),

    # ── Draft labels ─────────────────────────────────────────────────────────
    "draft_label":         ("📝 <b>Черновик #{id}:</b>", "📝 <b>Draft #{id}:</b>"),
    "draft_buttons_below": ("📝 <b>Черновик #{id}:</b> кнопки ниже", "📝 <b>Draft #{id}:</b> buttons below"),
    "draft_up":            ("⬆️ Черновик #{id}", "⬆️ Draft #{id}"),
    "draft_part":          ("📝 <b>Черновик #{id} (часть {n}/{total}):</b>", "📝 <b>Draft #{id} (part {n}/{total}):</b>"),

    # ── Publish / reject results ─────────────────────────────────────────────
    "post_published_toast": ("Опубликовано!", "Published!"),
    "post_published_full":  ("✅ <b>Пост #{id} опубликован в канале!</b>", "✅ <b>Post #{id} published!</b>"),
    "post_rejected_toast":  ("Отклонено.", "Rejected."),
    "post_rejected_full":   ("❌ Пост #{id} отклонён.", "❌ Post #{id} rejected."),
    "digest_rejected_toast":("Дайджест отклонён.", "Digest rejected."),
    "digest_rejected_full": ("❌ Дайджест #{id} отклонён.", "❌ Digest #{id} rejected."),
    "error_generic_toast":  ("Ошибка: {err}", "Error: {err}"),

    # ── Not found / unavailable ──────────────────────────────────────────────
    "post_not_found":       ("Пост не найден.", "Post not found."),
    "topic_not_found":      ("Тема не найдена.", "Topic not found."),
    "digest_not_found":     ("Дайджест не найден.", "Digest not found."),
    "variant_not_found":    ("Вариант не найден.", "Variant not found."),
    "img_prompt_unavailable":("Промпт для картинки недоступен.", "Image prompt not available."),
    "img_prompt_header":    ("🎨 <b>Промпт для картинки (пост #{id}):</b>", "🎨 <b>Image prompt (post #{id}):</b>"),
    "post_source":          ("ℹ️ <b>Источник поста #{id}:</b>\n", "ℹ️ <b>Post #{id} source:</b>\n"),

    # ── /status ──────────────────────────────────────────────────────────────
    "status_recent_digests": ("<b>Последние дайджесты:</b>", "<b>Recent digests:</b>"),
    "status_digest_line": (
        "{e} <b>Дайджест #{id}</b> [{status}] — {date} ({n} тем)",
        "{e} <b>Digest #{id}</b> [{status}] — {date} ({n} topics)",
    ),
    "status_recent_posts": ("<b>Последние посты:</b>", "<b>Recent posts:</b>"),
    "status_empty": ("Ничего нет.", "Nothing here yet."),
}


def t(key: str, **kwargs) -> str:
    """Return the string for `key` in the current language, formatted with kwargs."""
    pair = STRINGS.get(key)
    if pair is None:
        logger.warning("i18n: missing string key %r", key)
        return key
    text = pair[1] if get_language() == "en" else pair[0]
    return text.format(**kwargs) if kwargs else text
