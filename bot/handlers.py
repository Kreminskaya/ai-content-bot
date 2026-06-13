"""
aiogram 3.x router: all bot interactions with the admin.

Two main flows:

─── DIGEST FLOW (new) ─────────────────────────────────────────────────────────
Bot sends "📰 Дайджест за [Дата]" with 10 topic buttons + "❌ Отклонить всё".
When admin clicks a topic:
  1. Bot answers "⏳ Пишу пост по этой теме..."
  2. WriterCrew is triggered for that specific topic (returns to old publish flow).

─── POST APPROVAL FLOW (existing, triggered by WriterCrew) ────────────────────
  ┌─────────────────────────────────────┐
  │  ✅ Вар. 1   ✅ Вар. 2   ✅ Вар. 3  │  ← publish variant N directly
  │  🎨 Промпт для картинки             │  ← show image-gen prompt (if exists)
  │  ✏️ Редактировать   ❌ Отклонить    │
  └─────────────────────────────────────┘

Edit flow (FSM):
  waiting_text  → admin sends custom post text
  confirm_text  → admin confirms or cancels

Admin commands:
  /start   — welcome message
  /run     — trigger full digest pipeline manually
  /status  — last 5 posts and last 3 digests
"""

import asyncio
import logging
import re
from pathlib import Path

import httpx
from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    Chat,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    InputMediaVideo,
    LinkPreviewOptions,
    Message,
    MessageEntity,
)

def _html_to_entities(text: str, extra_entities: list[dict] | None = None) -> tuple[str, list[dict]]:
    """
    Convert basic HTML tags to Telegram MessageEntity-compatible dicts.
    Supports: <b>, <i>, <u>, <s>, <a href="...">, <code>.
    Returns (plain_text, entities_list_including_extra).
    """
    import html as _html
    import re

    entities: list[dict] = []
    offset = 0

    # Map HTML tag → Telegram entity type
    _TYPE_MAP = {
        "b": "bold",
        "strong": "bold",
        "i": "italic",
        "em": "italic",
        "u": "underline",
        "ins": "underline",
        "s": "strikethrough",
        "strike": "strikethrough",
        "del": "strikethrough",
        "code": "code",
        "pre": "pre",
    }

    # Pattern for self-closing tags like <br/>, <br>, </b>
    # Match <tag ...> optionally with /closing
    tag_pattern = re.compile(r'<(/?)(\w+)(?:\s+[^>]*)?>([^<]*)')

    stack: list[tuple[str, int, str | None]] = []  # (tag, start_offset, href_or_lang)
    result: list[str] = []

    i = 0
    while i < len(text):
        if text[i] == '<':
            # Try to match tag
            m = tag_pattern.match(text, i)
            if m:
                slash, tag, _ = m.groups()
                end = m.end()
                # Get the full interior of the tag (for href)
                full_tag = _html.unescape(text[i:end])
                if slash:
                    # closing tag
                    if stack and stack[-1][0] == tag:
                        _, start, href = stack.pop()
                        length = offset - start
                        ent: dict = {"type": _TYPE_MAP.get(tag, tag), "offset": start, "length": length}
                        if tag == "a" and href:
                            ent["type"] = "text_link"
                            ent["url"] = href
                        entities.append(ent)
                    i = end
                    continue
                else:
                    # opening tag
                    href: str | None = None
                    if tag == "a":
                        hm = re.search(r'href=["\']([^"\'>]+)', full_tag)
                        if hm:
                            href = _html.unescape(hm.group(1))
                    stack.append((tag, offset, href))
                    i = end
                    continue
            # Unmatched < — treat as literal
            result.append(text[i])
            offset += 1
            i += 1
        else:
            # Take characters up to next <
            next_lt = text.find('<', i)
            if next_lt == -1:
                chunk = text[i:]
                result.append(chunk)
                offset += len(chunk)
                break
            chunk = text[i:next_lt]
            result.append(chunk)
            offset += len(chunk)
            i = next_lt

    plain_text = "".join(result)

    # Append extra entities (custom emoji etc.)
    if extra_entities:
        entities.extend(extra_entities)

    return plain_text, entities

from config import (
    ADMIN_CHAT_ID, TARGET_CHANNEL_ID,
    CATEGORY_VISUAL, CATEGORY_LLM, CATEGORY_SERVICES, CATEGORY_ROBOTICS, CATEGORY_PRODUCTION,
    CATEGORY_VISUAL_RU, CATEGORY_LLM_RU, CATEGORY_SERVICES_RU, CATEGORY_ROBOTICS_RU, CATEGORY_PRODUCTION_RU,
    HARD_CATEGORIES, USEFUL_CATEGORIES, PRODUCTION_CATEGORY,
)
from bot import i18n
from bot.trail import trail_append, is_used
from database.models import (
    create_post,
    get_digest,
    get_post,
    get_recent_digests,
    get_recent_posts,
    set_digest_tg_message_id,
    set_tg_message_id,
    update_digest_status,
    update_status,
)

logger = logging.getLogger(__name__)
router = Router()

# ---------------------------------------------------------------------------
# Media-group buffer: collect messages sent as an album before processing
# ---------------------------------------------------------------------------
_media_group_buffer: dict[str, list[Message]] = {}
_media_group_lock = asyncio.Lock()
_media_group_tasks: dict[str, asyncio.Task] = {}
_MEDIA_GROUP_TIMEOUT = 3.0  # seconds to wait for all album messages

_media_download_dir = Path(__file__).resolve().parent.parent / "downloads"


def _friendly_llm_error(exc: Exception) -> str:
    """Convert provider exceptions into admin-friendly messages."""
    raw = str(exc)
    lowered = raw.lower()
    if "usage limits" in lowered or "rate limit" in lowered or "429" in lowered:
        return i18n.t("llm_limit")
    if "invalid_request_error" in lowered and "api usage limits" in lowered:
        return i18n.t("llm_limit_fallback")
    return i18n.t("writing_error", err=raw)


# ---------------------------------------------------------------------------
# Safety post-processor: strip any t.me links before displaying / publishing
# ---------------------------------------------------------------------------

def _delete_media_file(path: str | None) -> None:
    """Safely delete a downloaded media file after it's been published or rejected."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
        logger.info("Deleted media file: %s", path)
    except OSError as exc:
        logger.warning("Could not delete media file %s: %s", path, exc)


def _delete_media_files(paths: list[str] | None) -> None:
    """Delete all media files from a list (for album posts)."""
    for p in (paths or []):
        _delete_media_file(p)


def _get_media_paths(post: dict) -> list[str]:
    """Return list of valid existing media paths for a post (multi-media aware)."""
    paths = post.get("media_paths") or []
    if not paths:
        single = post.get("media_path")
        if single:
            paths = [single]
    return [p for p in paths if p and Path(p).exists()]


async def _send_media_group_or_single(
    bot: Bot,
    chat_id: int | str,
    paths: list[str],
    caption: str | None = None,
    parse_mode: str = "HTML",
) -> None:
    """
    Send one or more media files to chat_id.
    • 1 file  → send_photo / send_video
    • 2+ files → send_media_group (first item gets caption)
    """
    if not paths:
        return

    def _ext(p: str) -> str:
        return Path(p).suffix.lower()

    if len(paths) == 1:
        p = paths[0]
        if _ext(p) == ".mp4":
            await bot.send_video(chat_id=chat_id, video=FSInputFile(p),
                                 caption=caption, parse_mode=parse_mode)
        else:
            await bot.send_photo(chat_id=chat_id, photo=FSInputFile(p),
                                 caption=caption, parse_mode=parse_mode)
    else:
        # Build media group — first item carries caption
        media_group = []
        for i, p in enumerate(paths):
            item_caption = caption if i == 0 else None
            if _ext(p) == ".mp4":
                media_group.append(InputMediaVideo(
                    media=FSInputFile(p),
                    caption=item_caption,
                    parse_mode=parse_mode if item_caption else None,
                ))
            else:
                media_group.append(InputMediaPhoto(
                    media=FSInputFile(p),
                    caption=item_caption,
                    parse_mode=parse_mode if item_caption else None,
                ))
        await bot.send_media_group(chat_id=chat_id, media=media_group)


def _sanitize_post(text: str) -> str:
    """
    Last-mile safety net: remove Telegram channel links and @handles that
    would render as embedded channel cards in Telegram messages.
    Keeps t.me links that point to SPECIFIC POSTS (contain a message ID),
    as these are product/content links, not channel references.
    The primary stripping happens in agents/crew.py (_strip_tg_links).
    This covers text entered manually by the admin in the edit flow.
    """
    # Remove bare channel links: t.me/channel (no message ID after)
    text = re.sub(r'https?://t\.me/[A-Za-z0-9_]+/?\s', ' ', text)
    text = re.sub(r'https?://t\.me/[A-Za-z0-9_]+/?$', '', text)
    # Remove @username mentions
    text = re.sub(r'(?<!\S)@[A-Za-z][A-Za-z0-9_]{3,}(?!\S)', '', text)
    # Last-mile: strip any leaked pipeline markers (e.g. LLM forgot to strip them)
    text = re.sub(r'(?im)[^\n]*ФИНАЛЬНЫЙ\s+ПОСТ[^\n]*\n?', '', text)
    text = re.sub(r'(?im)[^\n]*ЧЕРНОВИК\s*:[^\n]*\n?', '', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


async def _upload_to_telegraph(file_path: str) -> str | None:
    """
    Upload a local image/video file to Telegraph and return its public URL.
    Returns None on any error (caller falls back to caption approach).
    """
    for attempt in range(2):
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            mime = "image/jpeg"
            suffix = path.suffix.lower()
            if suffix == ".png":
                mime = "image/png"
            elif suffix == ".gif":
                mime = "image/gif"
            elif suffix in (".mp4", ".mov"):
                mime = "video/mp4"
            async with httpx.AsyncClient(timeout=30) as client:
                with open(file_path, "rb") as f:
                    response = await client.post(
                        "https://telegra.ph/upload",
                        files={"file": (path.name, f, mime)},
                    )
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and data and "src" in data[0]:
                        return f"https://telegra.ph{data[0]['src']}"
        except Exception as exc:
            logger.warning("Telegraph upload attempt %d failed: %s", attempt + 1, exc)
            if attempt == 0:
                await asyncio.sleep(2)
    return None


# Only the admin can interact with this bot
router.message.filter(F.chat.id == ADMIN_CHAT_ID)
router.callback_query.filter(F.message.chat.id == ADMIN_CHAT_ID)


# ---------------------------------------------------------------------------
# FSM States
# ---------------------------------------------------------------------------

class EditState(StatesGroup):
    waiting_text = State()   # waiting for admin's custom post text
    confirm_text = State()   # admin is reviewing preview before publish


class ManualPostState(StatesGroup):
    waiting_content = State()  # waiting for raw news content from admin


# ---------------------------------------------------------------------------
# Digest keyboard builder
# ---------------------------------------------------------------------------

_CATEGORY_EMOJI = {
    CATEGORY_VISUAL:        "🎨",
    CATEGORY_LLM:           "🧠",
    CATEGORY_SERVICES:      "🛠",
    CATEGORY_ROBOTICS:      "🤖",
    CATEGORY_PRODUCTION:    "🎬",
    CATEGORY_VISUAL_RU:     "🎨",
    CATEGORY_LLM_RU:        "🧠",
    CATEGORY_SERVICES_RU:   "🛠",
    CATEGORY_ROBOTICS_RU:   "🤖",
    CATEGORY_PRODUCTION_RU: "🎬",
}


def digest_keyboard(digest_id: int, topics: list[dict]) -> InlineKeyboardMarkup:
    """
    Build the 10-button digest menu.
    Each topic gets its own button (one per row for readability).
    Last row: "❌ Отклонить всё" button.
    """
    rows: list[list[InlineKeyboardButton]] = []

    for topic in topics:
        idx = topic.get("index", 0)
        category = topic.get("category", "")
        emoji = _CATEGORY_EMOJI.get(category, "📌")
        title = topic.get("title", i18n.t("topic_fallback", idx=idx))
        # Truncate button label to fit Telegram's 64-char limit
        label = f"{emoji} {title}"
        if len(label) > 60:
            label = label[:57] + "…"

        rows.append([
            InlineKeyboardButton(
                text=label,
                callback_data=f"digest:topic:{digest_id}:{idx}",
            )
        ])

    rows.append([
        InlineKeyboardButton(
            text=i18n.t("btn_reject_all"),
            callback_data=f"digest:reject:{digest_id}",
        )
    ])

    return InlineKeyboardMarkup(inline_keyboard=rows)


# ---------------------------------------------------------------------------
# Post approval keyboard (4 buttons, used after WriterCrew)
# ---------------------------------------------------------------------------

def post_keyboard(post_id: int, has_image_prompt: bool = False) -> InlineKeyboardMarkup:
    """
    4-button keyboard for the final post approval flow:
      [✅ Постить]  [✏️ Редактировать]
      [❌ Отмена]   [🎨 Промпт]  ← only if image prompt exists
    """
    rows: list[list[InlineKeyboardButton]] = []

    rows.append([
        InlineKeyboardButton(
            text=i18n.t("btn_publish"),
            callback_data=f"post:pub:{post_id}:0",
        ),
        InlineKeyboardButton(
            text=i18n.t("btn_edit"),
            callback_data=f"post:edit:{post_id}",
        ),
    ])

    row2 = [
        InlineKeyboardButton(
            text=i18n.t("btn_cancel"),
            callback_data=f"post:reject:{post_id}",
        )
    ]
    rows.append(row2)

    return InlineKeyboardMarkup(inline_keyboard=rows)


def confirm_keyboard(post_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text=i18n.t("btn_publish"),
            callback_data=f"post:confirm:{post_id}",
        ),
        InlineKeyboardButton(
            text=i18n.t("btn_cancel"),
            callback_data=f"post:cancel_edit:{post_id}",
        ),
    ]])


def _edit_publish_keyboard(post_id: int) -> InlineKeyboardMarkup:
    """Edit-flow keyboard with consistent 'Постить' label."""
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text=i18n.t("btn_publish"),
            callback_data=f"post:confirm:{post_id}",
        ),
        InlineKeyboardButton(
            text=i18n.t("btn_cancel"),
            callback_data=f"post:cancel_edit:{post_id}",
        ),
    ]])


# ---------------------------------------------------------------------------
# send_digest_to_admin — called from scheduler after DigestCrew
# ---------------------------------------------------------------------------

async def send_digest_to_admin(bot: Bot, digest_id: int) -> None:
    """
    Send the "📰 Дайджест за [Дата]" message to admin with a 10-button menu.
    Each button represents one topic from the analyst's digest.
    """
    digest = get_digest(digest_id)
    if not digest:
        logger.error("send_digest_to_admin: digest #%s not found", digest_id)
        return

    topics: list[dict] = digest.get("topics", [])
    if not topics:
        logger.error("send_digest_to_admin: digest #%s has no topics", digest_id)
        return

    date = digest.get("date", "")
    lines: list[str] = [i18n.t("digest_header", date=date) + "\n"]

    # Split topics into sections by category
    hard       = [t for t in topics if t.get("category") in HARD_CATEGORIES]
    production = [t for t in topics if t.get("category") == CATEGORY_PRODUCTION]
    useful     = [t for t in topics if t.get("category") in USEFUL_CATEGORIES]

    if hard:
        lines.append(i18n.t("section_hard"))
        for t in hard:
            emoji = _CATEGORY_EMOJI.get(t.get("category", ""), "📌")
            lines.append(
                f"{emoji} <b>{t.get('index')}. {t.get('title', '')}</b>\n"
                f"<i>{t.get('summary', '')[:200]}</i>"
            )

    if production:
        lines.append("\n" + i18n.t("section_production"))
        for t in production:
            lines.append(
                f"🎬 <b>{t.get('index')}. {t.get('title', '')}</b>\n"
                f"<i>{t.get('summary', '')[:200]}</i>"
            )

    if useful:
        lines.append("\n" + i18n.t("section_useful"))
        for t in useful:
            emoji = _CATEGORY_EMOJI.get(t.get("category", ""), "📌")
            lines.append(
                f"{emoji} <b>{t.get('index')}. {t.get('title', '')}</b>\n"
                f"<i>{t.get('summary', '')[:200]}</i>"
            )

    total = len(topics)
    lines.append("\n<i>" + i18n.t("digest_footer", n=total) + "</i>")
    text = "\n\n".join(lines)

    msg = await bot.send_message(
        chat_id=ADMIN_CHAT_ID,
        text=text,
        reply_markup=digest_keyboard(digest_id, topics),
        parse_mode="HTML",
    )
    set_digest_tg_message_id(digest_id, msg.message_id)
    logger.info("Sent digest #%s to admin (tg msg %s)", digest_id, msg.message_id)


# ---------------------------------------------------------------------------
# send_post_to_admin — called after WriterCrew finishes for a topic
# ---------------------------------------------------------------------------

_TRUNC = 3500       # max chars for admin preview (Telegram message limit is 4096)


async def send_post_to_admin(bot: Bot, post_id: int) -> None:
    """
    Send the generated post draft to admin with a 4-button approval keyboard.

    Sending strategy:
      • text ≤ 1024 + media → single message: photo/video + full caption
      • text > 1024 + media → MSG 1: media without caption
                            → MSG 2: full text + keyboard (separate, no 1024 limit)
      • no media → single message: full text + keyboard

      [✅ Постить]  [✏️ Редактировать]
      [❌ Отмена]   [🎨 Промпт]
    """
    post = get_post(post_id)
    if not post:
        logger.error("send_post_to_admin: post #%s not found", post_id)
        return

    variants: list[str] = post.get("variants", [])
    if not variants:
        logger.error("send_post_to_admin: post #%s has no variants", post_id)
        return

    post_text: str   = _sanitize_post(variants[0])
    image_prompt     = post.get("image_prompt")
    media_paths_all  = _get_media_paths(post)
    keyboard         = post_keyboard(post_id, has_image_prompt=bool(image_prompt))

    # ── Send media with caption if text ≤ 1024, otherwise separate ───────────
    if media_paths_all:
        caption_text = i18n.t("draft_label", id=post_id) + f"\n\n{post_text}"
        if len(post_text) <= 1024:
            # Case 1: Short text + media → media group with caption
            try:
                await _send_media_group_or_single(
                    bot, ADMIN_CHAT_ID, media_paths_all,
                    caption=caption_text, parse_mode="HTML",
                )
                # send_media_group returns list of messages; attach keyboard separately
                if len(media_paths_all) > 1:
                    await bot.send_message(
                        chat_id=ADMIN_CHAT_ID,
                        text=i18n.t("draft_buttons_below", id=post_id),
                        parse_mode="HTML",
                        reply_markup=keyboard,
                    )
                else:
                    # Single media — keyboard was set in caption message, but aiogram
                    # send_photo/video doesn't auto-attach keyboard here; send separately
                    await bot.send_message(
                        chat_id=ADMIN_CHAT_ID,
                        text=i18n.t("draft_up", id=post_id),
                        parse_mode="HTML",
                        reply_markup=keyboard,
                    )
                logger.info(
                    "send_post_to_admin: sent %d media item(s) + keyboard for post #%s",
                    len(media_paths_all), post_id,
                )
                set_tg_message_id(post_id, 0)
                return
            except Exception as exc:
                logger.warning(
                    "send_post_to_admin: media send failed for post #%s (%s)",
                    post_id, exc,
                )
                # Fall through to text-only below
        else:
            # Case 2: Long text → media WITHOUT caption, then text separately
            try:
                await _send_media_group_or_single(
                    bot, ADMIN_CHAT_ID, media_paths_all,
                    caption=None,
                )
                logger.info(
                    "send_post_to_admin: sent %d media (no caption) for post #%s",
                    len(media_paths_all), post_id,
                )
            except Exception as exc:
                logger.warning(
                    "send_post_to_admin: media send failed for post #%s (%s)",
                    post_id, exc,
                )
            # Continue below: send full text + keyboard as a separate message

    # ── Send text + keyboard ─────────────────────────────────────────────────
    if len(post_text) <= _TRUNC:
        try:
            msg = await bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=i18n.t("draft_label", id=post_id) + f"\n\n{post_text}",
                reply_markup=keyboard,
                parse_mode="HTML",
            )
        except Exception as html_err:
            logger.warning(
                "send_post_to_admin: HTML parse failed for post #%s, "
                "falling back to plain text: %s", post_id, html_err,
            )
            # Fallback: strip HTML tags and send as plain text
            safe_text = re.sub(r"<[^>]+>", "", post_text)
            msg = await bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=i18n.t("draft_label", id=post_id) + f"\n\n{safe_text}",
                reply_markup=keyboard,
                parse_mode="HTML",
            )
        logger.info("Sent post #%s text to admin (tg msg %s)", post_id, msg.message_id)
        set_tg_message_id(post_id, msg.message_id)
    else:
        # Split long text - send first part with keyboard, rest without
        parts = _split_long_text(post_text, max_len=_TRUNC)
        if len(parts) == 1:
            try:
                msg = await bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=i18n.t("draft_label", id=post_id) + f"\n\n{parts[0]}",
                    reply_markup=keyboard,
                    parse_mode="HTML",
                )
            except Exception as html_err:
                logger.warning(
                    "send_post_to_admin: HTML parse failed for post #%s (single long part), "
                    "falling back to plain text: %s", post_id, html_err,
                )
                safe_text = re.sub(r"<[^>]+>", "", parts[0])
                msg = await bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=i18n.t("draft_label", id=post_id) + f"\n\n{safe_text}",
                    reply_markup=keyboard,
                    parse_mode="HTML",
                )
            logger.info("Sent post #%s text to admin (tg msg %s)", post_id, msg.message_id)
            set_tg_message_id(post_id, msg.message_id)
        else:
            # First part with keyboard
            try:
                msg = await bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=i18n.t("draft_part", id=post_id, n=1, total=len(parts)) + f"\n\n{parts[0]}",
                    reply_markup=keyboard,
                    parse_mode="HTML",
                )
            except Exception as html_err:
                logger.warning(
                    "send_post_to_admin: HTML parse failed for post #%s (split part 1), "
                    "falling back to plain text: %s", post_id, html_err,
                )
                safe_text = re.sub(r"<[^>]+>", "", parts[0])
                msg = await bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=i18n.t("draft_part", id=post_id, n=1, total=len(parts)) + f"\n\n{safe_text}",
                    reply_markup=keyboard,
                    parse_mode="HTML",
                )
            logger.info("Sent post #%s part 1/%d to admin (tg msg %s)", post_id, len(parts), msg.message_id)
            set_tg_message_id(post_id, msg.message_id)
            
            # Rest without keyboard
            for i, part in enumerate(parts[1:], start=2):
                await bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=part,
                    parse_mode="HTML",
                )
                if i <= len(parts):
                    await asyncio.sleep(0.2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_truncate_html(text: str, max_len: int = 1024) -> str:
    """
    Truncate text to max_len chars, safely closing any open HTML tags.
    Adds '…' at the end if truncated.
    """
    if len(text) <= max_len:
        return text
    
    # Leave room for '…' and potential closing tags
    truncated = text[:max_len - 20]
    
    # Find last safe break point (newline or space)
    break_point = truncated.rfind('\n')
    if break_point < max_len // 2:
        break_point = truncated.rfind(' ')
    if break_point > max_len // 2:
        truncated = truncated[:break_point]
    
    # Close any open HTML tags
    open_tags = re.findall(r'<(b|i|u|s|code|pre|blockquote)(?:\s[^>]*)?>', truncated)
    close_tags = re.findall(r'</(b|i|u|s|code|pre|blockquote)>', truncated)
    
    # Count unclosed tags
    open_count: dict[str, int] = {}
    for tag in open_tags:
        open_count[tag] = open_count.get(tag, 0) + 1
    for tag in close_tags:
        if tag in open_count and open_count[tag] > 0:
            open_count[tag] -= 1
    
    # Close unclosed tags in reverse order
    suffix = "…"
    for tag in reversed(open_tags):
        if open_count.get(tag, 0) > 0:
            suffix += f"</{tag}>"
            open_count[tag] -= 1
    
    return truncated + suffix


def _split_long_text(text: str, max_len: int = 4096) -> list[str]:
    """
    Safely split text into parts that fit within Telegram's message limit.
    Prefers splitting at newlines or spaces to avoid breaking words.
    """
    if len(text) <= max_len:
        return [text]

    parts = []
    while len(text) > max_len:
        # Try to find a good split point (newline preferred, then space)
        split_point = text.rfind('\n', 0, max_len)
        if split_point == -1 or split_point < max_len // 2:
            split_point = text.rfind(' ', 0, max_len)
        if split_point == -1:
            split_point = max_len
        
        part = text[:split_point].strip()
        if part:
            parts.append(part)
        text = text[split_point:].strip()

    if text:
        parts.append(text)

    return parts


# ---------------------------------------------------------------------------
# Publish helper
# ---------------------------------------------------------------------------

async def _publish(
    bot: Bot,
    post_id: int,
    text: str,
    call: CallbackQuery,
    entities: list[dict] | None = None,
    ) -> None:
    """
    Publish `text` to the target channel.

    New logic:
      1. If text ≤ 1024 and has media → send_photo/send_video with caption
      2. If text > 1024 and photo and media_url → telegraph method (hidden link preview)
      3. If text > 4096 → split into multiple messages safely
      4. If no media → send_message, split if needed

    Entities (custom emoji etc.) are passed straight to Telegram API when present;
    in that case parse_mode=HTML is NOT used because Telegram ignores entities
    when a parse_mode is set.
    """
    # Rebuild MessageEntity objects from the JSON-safe dicts stored in FSM
    entity_objs: list[MessageEntity] | None = None
    if entities:
        entity_objs = [MessageEntity(**e) for e in entities]
        # Keep edited text unmodified — entity offsets depend on exact character positions.
        # _sanitize_post strips t.me links / @handles, shifting offsets and breaking entities.
    else:
        text = _sanitize_post(text)

    async def _send_post_text(chat_id: int, _text: str, link_preview_options=None) -> None:
        """Send text with entities when available, otherwise with HTML parse_mode."""
        if entity_objs and len(_text) <= 4096:
            # Entities already present (user sent via Telegram formatting/custom emoji) — send as-is
            await bot.send_message(
                chat_id=chat_id,
                text=_text,
                entities=entity_objs,
                link_preview_options=link_preview_options,
            )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text=_text,
                parse_mode="HTML",
                link_preview_options=link_preview_options,
            )

    # Load media info from DB
    post = get_post(post_id)
    media_path: str | None = post.get("media_path") if post else None
    media_type: str | None = post.get("media_type") if post else None

    try:
        media_url: str | None = None

        # ── If entities present (edit flow), skip media — text-only publish ──
        if entity_objs:
            logger.info("Post #%s: entities present, text-only publish", post_id)
            # TRAIL: log send attempt
            trail_append(
                action="send",
                content_id=f"telegram:channel:{TARGET_CHANNEL_ID}:post:{post_id}",
                params={"post_id": post_id, "text_len": len(text), "parts": 1},
            )
            if len(text) <= 4096:
                await _send_post_text(chat_id=TARGET_CHANNEL_ID, _text=text)
            else:
                parts = _split_long_text(text)
                for i, part in enumerate(parts):
                    if i == 0 and entity_objs:
                        # First part: send with entities (offsets may be
                        # partially wrong after _split_long_text stripping,
                        # but entities near the start should survive)
                        await _send_post_text(chat_id=TARGET_CHANNEL_ID, _text=part)
                    else:
                        # Subsequent parts: send without entities
                        await bot.send_message(
                            chat_id=TARGET_CHANNEL_ID,
                            text=part,
                            parse_mode="HTML",
                        )
                    if i < len(parts) - 1:
                        await asyncio.sleep(0.3)
            trail_append(
                action="send",
                content_id=f"telegram:channel:{TARGET_CHANNEL_ID}:post:{post_id}",
                result={"status": "ok", "text_len": len(text)},
            )
            update_status(post_id, "published", final_text=text)
            _delete_media_file(media_path)
            await call.answer(i18n.t("post_published_toast"))
            _cid = f"telegram:chat:{ADMIN_CHAT_ID}:message:{call.message.message_id}"
            trail_append(action="edit", content_id=_cid, params={"post_id": post_id})
            try:
                await call.message.edit_text(
                    i18n.t("post_published_full", id=post_id) + "\n\n" + 
                    f"{text[:400]}{'…' if len(text) > 400 else ''}",
                    reply_markup=None,
                )
            except Exception:
                pass
            logger.info("Post #%s: text-only publish done (%d chars)", post_id, len(text))
            return

        # ── Step 1: try to get a public URL for the media ────────────────────
        if media_path and Path(media_path).exists():
            media_url = await _upload_to_telegraph(media_path)
            if media_url:
                logger.info("Post #%s: uploaded media to Telegraph: %s", post_id, media_url)
            else:
                logger.info("Post #%s: Telegraph upload failed", post_id)

        # ── Step 2: publish ──────────────────────────────────────────────────
        if media_url and len(text) <= 1024:
            # Case 1: Short text + media → single message with caption
            # caption max is 1024, but we already checked len(text) ≤ 1024
            if media_type == "photo":
                await bot.send_photo(
                    chat_id=TARGET_CHANNEL_ID,
                    photo=media_url,
                    caption=text,
                    caption_entities=entity_objs or None,
                    parse_mode=None if entity_objs else "HTML",
                )
            else:
                await bot.send_video(
                    chat_id=TARGET_CHANNEL_ID,
                    video=media_url,
                    caption=text,
                    caption_entities=entity_objs or None,
                    parse_mode=None if entity_objs else "HTML",
                )
            _cid = f"telegram:channel:{TARGET_CHANNEL_ID}:post:{post_id}"
            trail_append(action="send", content_id=_cid,
                params={"post_id": post_id, "text_len": len(text), "media_type": media_type},
                result={"status": "ok"})
            logger.info("Post #%s published with caption (%d chars)", post_id, len(text))

        elif media_url and len(text) > 1024:
            # Case 2: Long text + photo + media_url → telegraph hidden link preview
            if entity_objs and len(text) + len(media_url) + 40 > 4096:
                # Entity-safe path: skip hidden_link injection to keep offsets valid
                parts = _split_long_text(text, max_len=4096)
                for i, part in enumerate(parts):
                    await bot.send_message(
                        chat_id=TARGET_CHANNEL_ID,
                        text=part,
                        entities=entity_objs if i == 0 else None,
                        link_preview_options=LinkPreviewOptions(
                            is_disabled=False,
                            url=media_url,
                            prefer_large_media=True,
                            show_above_text=True,
                        ) if i == 0 else LinkPreviewOptions(is_disabled=True),
                    )
                    if i > 0 or len(parts) == 1:
                        await asyncio.sleep(0.3)
                _cid = f"telegram:channel:{TARGET_CHANNEL_ID}:post:{post_id}"
                trail_append(action="send", content_id=_cid,
                    params={"post_id": post_id, "text_len": len(text), "parts": len(parts), "entity_safe": True},
                    result={"status": "ok"})
                logger.info("Post #%s published entity-safe long (%d chars in %d parts)", post_id, len(text), len(parts))
            else:
                hidden_link = f'<a href="{media_url}">&#8205;</a>'
                total_len = len(text) + len(hidden_link)
                if total_len >= 4096:
                    # Single part with hidden link (edge case: text almost at limit)
                    await bot.send_message(
                        chat_id=TARGET_CHANNEL_ID,
                        text=hidden_link + text[:4096 - len(hidden_link)],
                        parse_mode="HTML",
                        link_preview_options=LinkPreviewOptions(
                            is_disabled=False,
                            url=media_url,
                            prefer_large_media=True,
                            show_above_text=True,
                        ),
                    )
                    _cid = f"telegram:channel:{TARGET_CHANNEL_ID}:post:{post_id}"
                    trail_append(action="send", content_id=_cid,
                        params={"post_id": post_id, "text_len": len(text), "hidden_link": True},
                        result={"status": "ok"})
                    logger.info("Post #%s published with hidden link preview (%d chars)", post_id, len(text))
                else:
                    # Case 3: Very long text → split and send with hidden link on first part
                    parts = _split_long_text(text, max_len=4096 - len(hidden_link))
                    for i, part in enumerate(parts):
                        if i == 0:
                            await bot.send_message(
                                chat_id=TARGET_CHANNEL_ID,
                                text=hidden_link + part,
                                parse_mode="HTML",
                                link_preview_options=LinkPreviewOptions(
                                    is_disabled=False,
                                    url=media_url,
                                    prefer_large_media=True,
                                    show_above_text=True,
                                ),
                            )
                        else:
                            await bot.send_message(
                                chat_id=TARGET_CHANNEL_ID,
                                text=part,
                                parse_mode="HTML",
                            )
                        if i < len(parts) - 1:
                            await asyncio.sleep(0.3)
                    _cid = f"telegram:channel:{TARGET_CHANNEL_ID}:post:{post_id}"
                    trail_append(action="send", content_id=_cid,
                        params={"post_id": post_id, "text_len": len(text), "parts": len(parts), "hidden_link": True},
                        result={"status": "ok"})
                    logger.info("Post #%s published with split (%d chars in %d parts)", post_id, len(text), len(parts))

        elif media_path and Path(media_path).exists():
            # Case 4: Local file, Telegraph failed
            #   • text ≤ 1024 → single message: photo + full caption
            #   • text > 1024 → media WITHOUT caption, then full text (split if needed) as reply
            if len(text) <= 1024:
                caption = text
                try:
                    if media_type == "photo":
                        await bot.send_photo(
                            chat_id=TARGET_CHANNEL_ID,
                            photo=FSInputFile(media_path),
                            caption=caption,
                            parse_mode="HTML",
                        )
                    else:
                        await bot.send_video(
                            chat_id=TARGET_CHANNEL_ID,
                            video=FSInputFile(media_path),
                            caption=caption,
                            parse_mode="HTML",
                        )
                    logger.info("Post #%s published with media+caption (%d chars)", post_id, len(text))
                except Exception as media_exc:
                    logger.warning("Post #%s: media+caption failed: %s — sending text only", post_id, media_exc)
                    if len(text) <= 4096:
                        await _send_post_text(chat_id=TARGET_CHANNEL_ID, _text=text)
                    else:
                        parts = _split_long_text(text)
                        for i, part in enumerate(parts):
                            await _send_post_text(chat_id=TARGET_CHANNEL_ID, _text=part)
                            if i < len(parts) - 1:
                                await asyncio.sleep(0.3)
            else:
                # Long text + local media → media first, then full text (as separate messages)
                try:
                    if media_type == "photo":
                        await bot.send_photo(
                            chat_id=TARGET_CHANNEL_ID,
                            photo=FSInputFile(media_path),
                            caption=None,
                        )
                    else:
                        await bot.send_video(
                            chat_id=TARGET_CHANNEL_ID,
                            video=FSInputFile(media_path),
                            caption=None,
                        )
                except Exception as media_exc:
                    logger.warning("Post #%s: media send failed: %s", post_id, media_exc)

                # Send full text (split if needed) — as a separate message, NOT reply
                if len(text) <= 4096:
                    await _send_post_text(chat_id=TARGET_CHANNEL_ID, _text=text)
                else:
                    parts = _split_long_text(text)
                    for i, part in enumerate(parts):
                        await _send_post_text(chat_id=TARGET_CHANNEL_ID, _text=part)
                        if i < len(parts) - 1:
                            await asyncio.sleep(0.3)
                logger.info("Post #%s published: media (no caption) + full text (%d chars)", post_id, len(text))

        else:
            # Case 5: No media → just text, split if needed
            if len(text) <= 4096:
                await _send_post_text(chat_id=TARGET_CHANNEL_ID, _text=text)
            else:
                parts = _split_long_text(text)
                for i, part in enumerate(parts):
                    await _send_post_text(chat_id=TARGET_CHANNEL_ID, _text=part)
                    if i < len(parts) - 1:
                        await asyncio.sleep(0.3)
                logger.info("Post #%s published (split into %d parts, %d chars)", post_id, len(parts), len(text))
            logger.info("Post #%s published to channel (text only, %d chars)", post_id, len(text))

        update_status(post_id, "published", final_text=text)
        _delete_media_file(media_path)   # free disk space after successful publish

    except Exception as exc:
        logger.exception("Failed to publish post #%s: %s", post_id, exc)
        update_status(post_id, "error")
        await call.answer(i18n.t("error_generic_toast", err=exc), show_alert=True)
        return

    # Confirm to admin — separate try so a failed edit never undoes the publish
    await call.answer(i18n.t("post_published_toast"))
    try:
        if entity_objs and len(text) <= 4096:
            await call.message.edit_text(  # type: ignore[union-attr]
                i18n.t("post_published_full", id=post_id) + "\n\n" + 
                f"{text[:400]}{'…' if len(text) > 400 else ''}",
                reply_markup=None,
            )
        else:
            await call.message.edit_text(  # type: ignore[union-attr]
                i18n.t("post_published_full", id=post_id) + "\n\n" + 
                f"<i>{text[:400]}{'…' if len(text) > 400 else ''}</i>",
                reply_markup=None,
                parse_mode="HTML",
            )
    except Exception as edit_exc:
        logger.warning("Post #%s: edit_text after publish failed (non-critical): %s", post_id, edit_exc)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(
        i18n.t("start_message"),
        parse_mode="HTML",
    )


@router.message(Command("post"))
async def cmd_post(message: Message, state: FSMContext) -> None:
    """Start the manual post creation flow."""
    await state.set_state(ManualPostState.waiting_content)
    await message.answer(
        i18n.t("manual_intro"),
        parse_mode="HTML",
    )


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    posts = get_recent_posts(5)
    digests = get_recent_digests(3)

    lines: list[str] = []

    if digests:
        digest_emoji = {"pending": "⏳", "processed": "✅", "rejected": "❌"}
        lines.append(i18n.t("status_recent_digests"))
        for d in digests:
            e = digest_emoji.get(d["status"], "❓")
            topic_count = len(d.get("topics", []))
            lines.append(
                i18n.t("status_digest_line", e=e, id=d["id"], status=d["status"], date=d["date"], n=topic_count)
            )

    if posts:
        emoji_map = {
            "pending":   "⏳",
            "published": "✅",
            "rejected":  "❌",
            "editing":   "✏️",
            "approved":  "👍",
            "error":     "🔴",
        }
        lines.append("\n" + i18n.t("status_recent_posts"))
        for p in posts:
            e = emoji_map.get(p["status"], "❓")
            img = " 🎨" if p.get("image_prompt") else ""
            lines.append(
                f"{e} <b>#{p['id']}</b> [{p['status']}]{img} — {p['created_at'][:16]}"
            )

    if not lines:
        await message.answer(i18n.t("status_empty"))
        return

    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("run"))
async def cmd_run(message: Message, bot: Bot) -> None:
    """Manually trigger the full digest pipeline."""
    from scheduler import run_pipeline_job  # local import to avoid circular dep

    await message.answer(i18n.t("running_pipeline"))
    try:
        await run_pipeline_job(bot)
    except Exception as exc:
        logger.exception("Manual pipeline run failed: %s", exc)
        await message.answer(
            i18n.t("run_error", err=exc), parse_mode="HTML"
        )


# ---------------------------------------------------------------------------
# Language: /language switcher + localized command menu
# ---------------------------------------------------------------------------

async def apply_command_menu(bot: Bot) -> None:
    """(Re)register the bot command menu in the current UI language."""
    from aiogram.types import BotCommand  # noqa: PLC0415
    await bot.set_my_commands([
        BotCommand(command="start",    description=i18n.t("cmd_start_desc")),
        BotCommand(command="run",      description=i18n.t("cmd_run_desc")),
        BotCommand(command="post",     description=i18n.t("cmd_post_desc")),
        BotCommand(command="status",   description=i18n.t("cmd_status_desc")),
        BotCommand(command="cancel",   description=i18n.t("cmd_cancel_desc")),
        BotCommand(command="language", description=i18n.t("cmd_language_desc")),
    ])


def _language_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=i18n.t("btn_lang_ru"), callback_data="lang:ru"),
        InlineKeyboardButton(text=i18n.t("btn_lang_en"), callback_data="lang:en"),
    ]])


@router.message(Command("language"))
async def cmd_language(message: Message) -> None:
    """Show the language picker (🇷🇺 / 🇬🇧)."""
    await message.answer(i18n.t("choose_language"), reply_markup=_language_keyboard())


@router.callback_query(F.data.startswith("lang:"))
async def cb_set_language(call: CallbackQuery) -> None:
    """Switch the whole interface + content language on the fly and remember it."""
    lang = call.data.split(":")[1]  # type: ignore[union-attr]
    i18n.set_language(lang)
    await call.answer()
    # Re-register the command menu in the new language, then confirm in the new language.
    try:
        await apply_command_menu(call.bot)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not update command menu after language switch: %s", exc)
    try:
        await call.message.edit_text(i18n.t("language_set"))  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        await call.message.answer(i18n.t("language_set"))  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Callback: Digest topic selected → run WriterCrew
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("digest:topic:"))
async def cb_digest_topic(call: CallbackQuery) -> None:
    """
    Admin clicked on a digest topic button.
    1. Acknowledge immediately with i18n.t("writing_post_toast")
    2. Trigger WriterCrew for this specific topic.
    3. Send the resulting post variants back for approval.
    """
    parts = call.data.split(":")  # type: ignore[union-attr]
    # digest:topic:{digest_id}:{topic_index}
    digest_id   = int(parts[2])
    topic_index = int(parts[3])

    digest = get_digest(digest_id)
    if not digest:
        await call.answer(i18n.t("digest_not_found"), show_alert=True)
        return

    topics: list[dict] = digest.get("topics", [])
    topic = next((t for t in topics if t.get("index") == topic_index), None)
    if not topic:
        await call.answer(i18n.t("topic_not_found"), show_alert=True)
        return

    await call.answer(i18n.t("writing_post_toast"))
    await call.message.reply(  # type: ignore[union-attr]
        i18n.t("writing_on_topic") + "\n\n" + 
        f"<b>{topic.get('title', '')}</b>\n"
        f"<i>{topic.get('summary', '')}</i>",
        parse_mode="HTML",
    )

    # Mark digest as processed
    update_digest_status(digest_id, "processed")

    # Run WriterCrew for this topic
    bot: Bot = call.bot  # type: ignore[assignment]
    try:
        await _run_writer_for_topic(bot, topic)
    except Exception as exc:
        logger.exception("WriterCrew failed for topic %s: %s", topic_index, exc)
        await bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=_friendly_llm_error(exc),
            parse_mode="HTML",
        )


async def _run_writer_for_topic(bot: Bot, topic: dict) -> None:
    """
    Run the full writer pipeline (WriterCrew) for a single digest topic
    and send TWO messages to admin:

      MSG 1 — ℹ️ Служебное: ссылка на оригинальный источник (только для админа).
      MSG 2 — Черновик: скачанное медиа (если есть) + текст поста + кнопки.

    Media is downloaded via Telethon as a clean file — no forward/copy,
    no source channel trace in the published result.
    """
    from agents.crew import run_pipeline                          # noqa: PLC0415
    from agents.reviews_crew import run_reviews                   # noqa: PLC0415
    from parsers.telegram_userbot import download_media_from_url  # noqa: PLC0415

    source_url: str  = topic.get("url", "")
    topic_title: str = topic.get("title", "")

    # ── Step 1a: Collect user reviews (parallel with nothing — runs first) ───
    # Hard-capped at 15s internally; returns empty ReviewsResult on any failure.
    reviews_result = await run_reviews(topic_title, source_url)
    if reviews_result.summary:
        logger.info(
            "_run_writer_for_topic: ReviewsAgent found %d entries for '%s'",
            len(reviews_result.reviews), topic_title,
        )
    else:
        logger.info(
            "_run_writer_for_topic: ReviewsAgent found nothing for '%s', continuing without",
            topic_title,
        )

    # ── Step 1b: Run WriterCrew ──────────────────────────────────────────────
    # Step 1.5: Check if source has media BEFORE running WriterCrew
    _has_source_media = False
    _media_type_hint = None
    _dl_path = None

    # Priority 1: use media pre-downloaded by parser during fetch
    _prefetched = topic.get("media_path", "") or ""
    if _prefetched and Path(_prefetched).exists():
        _dl_path = _prefetched
        _has_source_media = True
        ext = Path(_prefetched).suffix.lower()
        _media_type_hint = "video" if ext in (".mp4", ".mov", ".avi") else "photo"
        logger.info("Using pre-fetched media: %s (%s)", _dl_path, _media_type_hint)

    # Priority 2: Bot API forward (works for channels that allow forwarding)
    if not _has_source_media and source_url and "t.me/" in source_url:
        try:
            _tme_match = re.match(
                r"https?://t\.me/([A-Za-z0-9_]+)/(\d+)", source_url
            )
            if _tme_match:
                _ch, _mid = _tme_match.group(1), int(_tme_match.group(2))
                _fwd = await bot.forward_message(
                    chat_id=ADMIN_CHAT_ID,
                    from_chat_id=f"@{_ch}",
                    message_id=_mid,
                    disable_notification=True,
                )
                _fid: str | None = None
                _fext = ".jpg"
                if _fwd.photo:
                    _fid = _fwd.photo[-1].file_id
                    _fext = ".jpg"
                    _media_type_hint = "photo"
                elif _fwd.video:
                    _fid = _fwd.video.file_id
                    _fext = ".mp4"
                    _media_type_hint = "video"
                elif _fwd.document:
                    _fid = _fwd.document.file_id
                    _fext = Path(_fwd.document.file_name or "doc").suffix or ".bin"
                    _media_type_hint = "document"
                if _fid:
                    _media_download_dir.mkdir(parents=True, exist_ok=True)
                    _tg_file = await bot.get_file(_fid)
                    _local = _media_download_dir / f"digest_{_mid}{_fext}"
                    await bot.download_file(_tg_file.file_path, destination=_local)
                    _dl_path = str(_local)
                    _has_source_media = True
                    logger.info("Bot API media fallback: %s (%s)", _dl_path, _media_type_hint)
                await bot.delete_message(chat_id=ADMIN_CHAT_ID, message_id=_fwd.message_id)
        except Exception as _exc:
            logger.warning("Bot API media fallback failed for %s: %s", source_url, _exc)

    news_items = [{
        "source":     source_url,
        "text":       (
            f"{topic_title}\n\n"
            f"{topic.get('summary', '')}\n\n"
            f"{topic.get('raw_text', '')}"
        ).strip(),
        "date":       "",
        "url":        source_url,
        "media_path": _dl_path if _has_source_media else None,
        "has_media":  _has_source_media,
        "media_type": _media_type_hint,
    }]

    # Последние опубликованные посты для RhythmChecker
    recent_rows = get_recent_posts(limit=10)
    recent_published = [
        r for r in recent_rows
        if r.get("status") == "published" and r.get("final_text")
    ]
    recent_posts_str = "\n---\n".join(r["final_text"] for r in recent_published[:7])

    result = await run_pipeline(
        news_items,
        recent_posts=recent_posts_str,
        user_reviews=reviews_result.summary,
    )

    if not result.variants:
        await bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=i18n.t("no_variants"),
        )
        return

    # ── Step 2: Use pre-downloaded media (already fetched before WriterCrew) ──
    media_path: str | None = _dl_path if _has_source_media else None
    media_type: str | None = _media_type_hint if _has_source_media else None

    # ── Step 3: Save to DB ───────────────────────────────────────────────────
    post_id = create_post(
        source_urls=[source_url],
        raw_news=f"{topic.get('title', '')}: {topic.get('summary', '')}",
        variants=result.variants,
        image_prompt=result.image_prompt,
        researcher_summary=result.researcher_summary,
        media_path=media_path,
        media_type=media_type,
    )

    # ── Step 4: MSG 1 — Service message (source info, admin-only) ───────────
    if source_url:
        await bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=(
                i18n.t("post_source", id=post_id) + 
                f"{source_url}"
            ),
            parse_mode="HTML",
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )

    # ── Step 5: MSG 2 — Draft (media + text + keyboard) ─────────────────────
    await send_post_to_admin(bot, post_id)


# ---------------------------------------------------------------------------
# Callback: Reject all digest topics
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("digest:reject:"))
async def cb_digest_reject(call: CallbackQuery) -> None:
    digest_id = int(call.data.split(":")[-1])  # type: ignore[union-attr]
    update_digest_status(digest_id, "rejected")
    await call.answer(i18n.t("digest_rejected_toast"))
    try:
        await call.message.edit_text(  # type: ignore[union-attr]
            i18n.t("digest_rejected_full", id=digest_id),
            reply_markup=None,
        )
    except Exception:
        pass
    logger.info("Digest #%s rejected by admin", digest_id)


# ---------------------------------------------------------------------------
# Callback: Publish a variant directly
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("post:pub:"))
async def cb_publish_variant(call: CallbackQuery) -> None:
    _, _, post_id_str, variant_idx_str = call.data.split(":")  # type: ignore[union-attr]
    post_id     = int(post_id_str)
    variant_idx = int(variant_idx_str)

    post = get_post(post_id)
    if not post:
        await call.answer(i18n.t("post_not_found"), show_alert=True)
        return

    # ── Guard: if post is being edited, block direct publish ──────────────
    if post.get("status") == "editing":
        await call.answer(
            i18n.t("edit_in_progress"),
            show_alert=True,
        )
        return

    variants: list[str] = post.get("variants", [])
    if variant_idx >= len(variants):
        await call.answer(i18n.t("variant_not_found"), show_alert=True)
        return

    update_status(post_id, "approved", selected_variant=variant_idx)
    await _publish(
        bot=call.bot,  # type: ignore[arg-type]
        post_id=post_id,
        text=variants[variant_idx],
        call=call,
    )


# ---------------------------------------------------------------------------
# Callback: Show image prompt
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("post:imgprompt:"))
async def cb_image_prompt(call: CallbackQuery) -> None:
    post_id = int(call.data.split(":")[-1])  # type: ignore[union-attr]
    post = get_post(post_id)
    if not post:
        await call.answer(i18n.t("post_not_found"), show_alert=True)
        return

    prompt = post.get("image_prompt") or ""
    if not prompt:
        await call.answer(i18n.t("img_prompt_unavailable"), show_alert=True)
        return

    await call.message.reply(  # type: ignore[union-attr]
        i18n.t("img_prompt_header", id=post_id) + "\n\n"
        f"<code>{prompt}</code>",
        parse_mode="HTML",
    )
    await call.answer()


# ---------------------------------------------------------------------------
# Callback: Reject post variants
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("post:reject:"))
async def cb_reject(call: CallbackQuery) -> None:
    post_id = int(call.data.split(":")[-1])  # type: ignore[union-attr]
    update_status(post_id, "rejected")
    # Free disk space for rejected posts
    post = get_post(post_id)
    if post:
        _delete_media_file(post.get("media_path"))
    # answer first — always — so Telegram stops the spinner
    await call.answer(i18n.t("post_rejected_toast"))
    # message may be a photo/video (no edit_text), try gracefully
    try:
        await call.message.edit_text(  # type: ignore[union-attr]
            i18n.t("post_rejected_full", id=post_id),
            reply_markup=None,
        )
    except Exception:
        try:
            await call.message.edit_caption(  # type: ignore[union-attr]
                caption=i18n.t("post_rejected_full", id=post_id),
                reply_markup=None,
            )
        except Exception:
            pass  # message already gone or uneditable — that's fine


# ---------------------------------------------------------------------------
# Callback: Start edit flow
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("post:edit:"))
async def cb_edit_start(call: CallbackQuery, state: FSMContext) -> None:
    post_id = int(call.data.split(":")[-1])  # type: ignore[union-attr]
    post = get_post(post_id)
    if not post:
        await call.answer(i18n.t("post_not_found"), show_alert=True)
        return

    update_status(post_id, "editing")
    await state.set_state(EditState.waiting_text)
    await state.update_data(post_id=post_id)

    # ── Disable buttons on the original draft message ─────────────────────
    # Prevent admin from clicking "✅ Постить" on the original message,
    # which would publish the AI-generated original instead of the edited text.
    try:
        await call.message.edit_reply_markup(  # type: ignore[union-attr]
            reply_markup=None,   # removes the keyboard entirely
        )
    except Exception as e:
        logger.warning(
            "cb_edit_start: edit_reply_markup failed for post #%s: %s. "
            "Original keyboard may still be visible — guard in cb_publish_variant will block it.",
            post_id, e,
        )
        # Notify admin to use the confirm message's button
        await call.answer(
            i18n.t("use_new_confirm"),
            show_alert=True,
        )

    variants: list[str] = post.get("variants", [])
    reference = variants[0] if variants else ""

    await call.message.reply(  # type: ignore[union-attr]
        i18n.t("edit_mode"),
        parse_mode="HTML",
    )

    if reference:
        await call.message.answer(  # type: ignore[union-attr]
            reference,
            parse_mode="HTML",
        )

    await call.answer()


@router.message(Command("cancel"), ManualPostState.waiting_content)
async def cmd_cancel_manual(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(i18n.t("manual_cancelled"))


@router.message(Command("cancel"), EditState.waiting_text)
@router.message(Command("cancel"), EditState.confirm_text)
async def cmd_cancel_edit(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    if post_id := data.get("post_id"):
        update_status(post_id, "pending")
    await state.clear()
    await message.answer(i18n.t("edit_cancelled"))


@router.message(EditState.waiting_text)
async def fsm_receive_text(message: Message, state: FSMContext) -> None:
    raw = message.text or ""
    if not raw.strip():
        await message.answer(i18n.t("text_empty"))
        return

    data = await state.get_data()
    post_id = data["post_id"]

    # html_text preserves all formatting (bold, italic, links) as HTML tags.
    # Also serialize entities separately for the userbot path (premium emoji).
    html_text = message.html_text
    entities = message.entities or []
    custom_emoji_count = sum(1 for e in entities if e.type == "custom_emoji")
    entities_data: list[dict] | None = None
    if entities:
        try:
            entities_data = [e.model_dump(exclude_none=True) for e in entities]
        except Exception:
            entities_data = None

    logger.info(
        "fsm_receive_text post #%s: len=%d entities=%d custom_emoji=%d",
        post_id, len(raw), len(entities), custom_emoji_count,
    )

    await state.set_state(EditState.confirm_text)
    await state.update_data(
        edited_text=html_text,      # HTML for parse_mode="HTML" publish (preserves bold/italic)
        edited_entities=entities_data,  # raw entities for userbot path (preserves premium emoji)
        src_chat_id=message.chat.id,
        src_message_id=message.message_id,
    )

    await message.answer(
        i18n.t("text_saved"),
        reply_markup=_edit_publish_keyboard(post_id),
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# Callback: Confirm edited text → publish
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("post:confirm:"), EditState.confirm_text)
async def cb_confirm_edit(call: CallbackQuery, state: FSMContext) -> None:
    post_id = int(call.data.split(":")[-1])  # type: ignore[union-attr]
    data = await state.get_data()
    edited_text: str = data.get("edited_text", "")
    edited_entities: list[dict] | None = data.get("edited_entities")
    src_chat_id: int | None = data.get("src_chat_id")
    src_message_id: int | None = data.get("src_message_id")

    await state.clear()

    if not edited_text:
        await call.answer(i18n.t("text_is_empty"), show_alert=True)
        return

    import json  # noqa: PLC0415
    from database.models import get_post, update_status  # noqa: PLC0415

    post = get_post(post_id)
    if post and post.get("variants"):
        variants = post["variants"]
        if variants:
            variants[0] = edited_text
            update_status(post_id, "approved",
                          selected_variant=0,
                          variants=json.dumps(variants, ensure_ascii=False))

    has_custom_emoji = edited_entities and any(
        e.get("type") == "custom_emoji" for e in edited_entities
    )
    logger.info(
        "cb_confirm_edit post #%s: text_len=%d has_custom_emoji=%s",
        post_id, len(edited_text), has_custom_emoji,
    )

    if has_custom_emoji:
        # Bot API cannot send custom/premium emoji to channels — use MTProto userbot.
        from parsers.telegram_userbot import publish_to_channel  # noqa: PLC0415
        ok = await publish_to_channel(
            channel_id=TARGET_CHANNEL_ID,
            text=edited_text,
            entities=edited_entities,  # type: ignore[arg-type]
        )
        if ok:
            update_status(post_id, "published", final_text=edited_text)
            await call.answer(i18n.t("post_published_toast"))
            await call.message.edit_text(  # type: ignore[union-attr]
                i18n.t("post_published_full", id=post_id),
                reply_markup=None,
                parse_mode="HTML",
            )
            return
        # Fallback if userbot failed — send via bot API with HTML (preserves formatting,
        # but premium emoji will render as plain since Bot API can't do custom emoji in channels).
        logger.warning("cb_confirm_edit: userbot failed, falling back to bot API (HTML mode)")

    # edited_text is html_text → pass entities=None so _publish uses parse_mode="HTML".
    # Mixing html_text with entities= would send <b> tags as literal text.
    await _publish(
        bot=call.bot,  # type: ignore[arg-type]
        post_id=post_id,
        text=edited_text,
        call=call,
        entities=None,
    )


@router.callback_query(F.data.startswith("post:cancel_edit:"))
async def cb_cancel_edit_btn(call: CallbackQuery, state: FSMContext) -> None:
    post_id = int(call.data.split(":")[-1])  # type: ignore[union-attr]
    await state.clear()
    update_status(post_id, "pending")
    await call.message.edit_text(i18n.t("edit_cancelled"), reply_markup=None)  # type: ignore[union-attr]
    await call.answer()


# ---------------------------------------------------------------------------
# Manual post flow: /post → receive raw content → WriterCrew → draft
# ---------------------------------------------------------------------------

@router.message(ManualPostState.waiting_content)
async def fsm_receive_manual_content(message: Message, state: FSMContext) -> None:
    """
    Receive any message from admin (text, forward, link) as raw news material.
    Passes it to WriterCrew and sends back a draft with the approval keyboard.

    Media-group handling: when Telegram sends an album (2+ photos/videos),
    messages arrive separately with the same media_group_id. We buffer them,
    wait for the full group, then merge text + download media before processing.
    """
    # ── Media-group: buffer and wait for all album messages ────────────
    if message.media_group_id:
        group_id = message.media_group_id
        async with _media_group_lock:
            if group_id not in _media_group_buffer:
                _media_group_buffer[group_id] = []
            _media_group_buffer[group_id].append(message)

            # Cancel previous timer for this group (resets the countdown)
            prev_task = _media_group_tasks.pop(group_id, None)
            if prev_task and not prev_task.done():
                prev_task.cancel()

            # Start new timer — when it fires, the group is considered complete
            _media_group_tasks[group_id] = asyncio.create_task(
                _process_media_group(group_id, message.bot, state)  # type: ignore[arg-type]
            )
        return  # Don't process immediately — wait for timer

    # ── Single message: process normally ───────────────────────────────
    await state.clear()

    # ── Extract raw text content ────────────────────────────────────────
    raw_text = (message.text or message.caption or "").strip()

    # If admin forwarded a message without text (e.g. media-only), note it
    if not raw_text and message.forward_origin:
        raw_text = "[Forwarded message without text]"

    if not raw_text:
        await message.answer(
            i18n.t("manual_no_text"),
        )
        return

    # ── Extract any URL from entities ────────────────────────────────────────
    source_url = ""
    entities = message.entities or message.caption_entities or []
    for ent in entities:
        if ent.type == "url":
            candidate = raw_text[ent.offset: ent.offset + ent.length]
            # Prefer non-Telegram URLs for factual context
            if "t.me/" not in candidate:
                source_url = candidate
                break
            elif not source_url:
                source_url = candidate  # fallback: use TG link if nothing else

    await message.answer(i18n.t("writing_post"), parse_mode="HTML")

    # ── Download media from the forwarded/attached message (Bot API) ────
    # Bot API works fine here — no Telethon/CDN limitations.
    # Limit: Bot API max file size = 20 MB (covers most news photos/videos).
    single_media: list[str] = []
    _media_download_dir.mkdir(parents=True, exist_ok=True)
    _file_id: str | None = None
    _ext = ".jpg"
    if message.photo:
        _file_id = message.photo[-1].file_id
        _ext = ".jpg"
    elif message.video:
        _file_id = message.video.file_id
        _ext = ".mp4"
    elif message.document:
        _file_id = message.document.file_id
        _doc_name = message.document.file_name or "document"
        _ext = Path(_doc_name).suffix or ".bin"
    if _file_id:
        try:
            _tg_file = await message.bot.get_file(_file_id)  # type: ignore[union-attr]
            _local = _media_download_dir / f"manual_{message.message_id}{_ext}"
            await message.bot.download_file(_tg_file.file_path, destination=_local)  # type: ignore[union-attr]
            single_media.append(str(_local))
            logger.info("Manual single: downloaded media %s", _local)
        except Exception as _exc:
            logger.warning("Manual single: media download failed: %s", _exc)

    bot: Bot = message.bot  # type: ignore[assignment]
    try:
        await _run_writer_for_manual(bot, raw_text, source_url,
                                     media_paths=single_media if single_media else None)
    except Exception as exc:
        logger.exception("Manual WriterCrew failed: %s", exc)
        await bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=i18n.t("writing_error", err=exc),
            parse_mode="HTML",
        )


async def _process_media_group(
    group_id: str,
    bot: Bot,
    state: FSMContext,
) -> None:
    """Wait for all media-group messages to arrive, then merge and process."""
    await asyncio.sleep(_MEDIA_GROUP_TIMEOUT)

    async with _media_group_lock:
        messages = _media_group_buffer.pop(group_id, [])
        _media_group_tasks.pop(group_id, None)

    if not messages:
        await state.clear()
        return

    # Sort by message_id to preserve order
    messages.sort(key=lambda m: m.message_id)

    # ── Merge text from all messages ──────────────────────────────────
    raw_text_parts: list[str] = []
    source_url = ""
    for msg in messages:
        text = (msg.text or msg.caption or "").strip()
        if text:
            raw_text_parts.append(text)

        # Extract source URL from entities
        entities = msg.entities or msg.caption_entities or []
        for ent in entities:
            if ent.type == "url":
                candidate = text[ent.offset: ent.offset + ent.length]
                if "t.me/" not in candidate:
                    source_url = candidate
                    break
                elif not source_url:
                    source_url = candidate

    raw_text = "\n\n".join(raw_text_parts)

    # If forward without text but with media, note it
    if not raw_text and any(m.forward_origin for m in messages):
        raw_text = "[Forwarded message with media, no text]"

    if not raw_text:
        await state.clear()
        # Send error to the FIRST message's chat
        first_msg = messages[0]
        await bot.send_message(
            chat_id=first_msg.chat.id,
            text=i18n.t("manual_no_text"),
        )
        return

    # ── Download all media files ──────────────────────────────────────
    media_files: list[str] = []
    _media_download_dir.mkdir(parents=True, exist_ok=True)

    for msg in messages:
        file_id: str | None = None
        ext = ".jpg"

        if msg.photo:
            # Take largest photo (last in array)
            file_id = msg.photo[-1].file_id
            ext = ".jpg"
        elif msg.video:
            file_id = msg.video.file_id
            ext = ".mp4"
        elif msg.document:
            file_id = msg.document.file_id
            doc_name = msg.document.file_name or "document"
            ext = Path(doc_name).suffix or ".bin"

        if not file_id:
            continue

        try:
            tg_file = await bot.get_file(file_id)
            local_path = _media_download_dir / f"mg_{group_id}_{msg.message_id}{ext}"
            await bot.download_file(tg_file.file_path, destination=local_path)
            media_files.append(str(local_path))
            logger.info(
                "_process_media_group: downloaded %s (%s bytes)",
                local_path, local_path.stat().st_size if local_path.exists() else 0,
            )
        except Exception as exc:
            logger.warning(
                "_process_media_group: failed to download media from msg %d: %s",
                msg.message_id, exc,
            )

    await state.clear()

    # ── Notify admin and run pipeline ─────────────────────────────────
    if raw_text == "[Forwarded message with media, no text]":
        await bot.send_message(
            chat_id=messages[0].chat.id,
            text=i18n.t("writing_post") + "\n\n" + 
                 i18n.t("media_received", n=len(media_files)),
            parse_mode="HTML",
        )
    else:
        await bot.send_message(
            chat_id=messages[0].chat.id,
            text=i18n.t("writing_post"),
            parse_mode="HTML",
        )

    try:
        await _run_writer_for_manual(
            bot, raw_text, source_url, media_paths=media_files if media_files else None,
        )
    except Exception as exc:
        logger.exception("Manual WriterCrew failed for media_group %s: %s", group_id, exc)
        await bot.send_message(
            chat_id=messages[0].chat.id,
            text=i18n.t("writing_error", err=exc),
            parse_mode="HTML",
        )


async def _run_writer_for_manual(
    bot: Bot,
    raw_content: str,
    source_url: str = "",
    media_paths: list[str] | None = None,
) -> None:
    """
    Run WriterCrew for manually provided raw news content.

    Works identically to _run_writer_for_topic but:
      • input comes directly from admin, not from DigestCrew;
      • no service-message with source link (admin already knows the source);
      • no media download (admin can add media manually if needed).

    All ToV rules (no hashtags, no TG-channel mentions, no hallucinations, etc.)
    are enforced via the same prompts as in the digest flow.

    If media_paths is provided, the first file is used as the primary media.
    """
    from agents.crew import run_pipeline                # noqa: PLC0415

    # Build a synthetic news_items entry with a manual-mode marker so the LLM
    # knows this content was hand-picked by the channel author.
    tg_source_url = source_url if source_url and "t.me/" in source_url else ""
    ext_source_url = source_url if source_url and "t.me/" not in source_url else ""

    news_items = [{
        "source":     "manual news from the channel author",
        "text":       (
            "[MANUAL MODE]\n"
            "The channel author sent this news directly for publishing.\n"
            "Write a post following the same style rules as for the digest.\n\n"
            + raw_content
        ),
        "date":       "",
        # Pass URL for factual context (не для вставки в пост).
        "url":        ext_source_url or tg_source_url,
        "media_path": None,
    }]

    # Последние опубликованные посты для RhythmChecker
    recent_rows = get_recent_posts(limit=10)
    recent_published = [
        r for r in recent_rows
        if r.get("status") == "published" and r.get("final_text")
    ]
    recent_posts_str = "\n---\n".join(r["final_text"] for r in recent_published[:7])

    result = await run_pipeline(news_items, recent_posts=recent_posts_str)

    if not result.variants:
        await bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=i18n.t("no_variants"),
        )
        return

    # Determine primary media path and type
    primary_media: str | None = None
    media_type: str | None = None
    if media_paths:
        primary_media = media_paths[0]
        ext = Path(primary_media).suffix.lower() if primary_media else ""
        if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            media_type = "photo"
        elif ext in (".mp4", ".mov", ".avi"):
            media_type = "video"
        elif media_paths:
            media_type = "document"

    post_id = create_post(
        source_urls=[source_url] if source_url else [],
        raw_news=raw_content[:500],
        variants=result.variants,
        image_prompt=result.image_prompt,
        researcher_summary=result.researcher_summary,
        media_path=primary_media,
        media_type=media_type,
    )

    if media_paths and len(media_paths) > 1:
        logger.info(
            "_run_writer_for_manual: %d media files, primary=%s",
            len(media_paths), primary_media,
        )

    # Single message: draft + keyboard (no service message needed —
    # admin already knows where the news came from)
    await send_post_to_admin(bot, post_id)
