import asyncio
"""
Telethon userbot — reads Telegram channels as a regular user via MTProto.

Why Telethon instead of HTTP web-preview (t.me/s/):
  • Works for any public channel, not only those with web-preview enabled
  • Returns full post history up to the requested limit
  • Downloads media (photos, videos, documents) with the actual file
  • Significantly more reliable and faster than HTML-scraping

First-time setup:
  Fill in TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_PHONE in .env,
  then run once interactively:

      python auth_userbot.py

  This creates `userbot.session` in the project root.
  After that every run reuses the saved session — no code entry needed.

Graceful degradation:
  • If credentials are missing → is_available == False → caller falls back to HTTP
  • If session file is absent → logs an error, returns [] → caller falls back to HTTP
  • If a channel is private or doesn't exist → logged, skipped, other channels proceed
  • Media download failure → logged, post still returned without media_path

Output format (each post dict):
  {
    "channel":    str,           # @username of the source channel
    "message_id": int,
    "text":       str,           # post body (or "[media without text]")
    "date":       datetime,      # timezone-aware UTC datetime
    "media_path": str | None,    # absolute path to downloaded file, or None
    "link":       str,           # permanent t.me link
    # ── normalised aliases for downstream compatibility ──
    "source":     str,           # https://t.me/<channel>
    "url":        str,           # same as link
  }
"""

import logging
import re
from datetime import datetime
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import (
    ChannelPrivateError,
    FloodWaitError,
    UsernameInvalidError,
    UsernameNotOccupiedError,
)
from telethon.tl.types import (
    MessageEntityBold,
    MessageEntityBlockquote,
    MessageEntityCode,
    MessageEntityCustomEmoji,
    MessageEntityItalic,
    MessageEntityPre,
    MessageEntitySpoiler,
    MessageEntityStrike,
    MessageEntityTextUrl,
    MessageEntityUnderline,
)

from config import (
    MEDIA_DIR,
    TELEGRAM_API_HASH,
    TELEGRAM_API_ID,
    TELEGRAM_PHONE,
    USERBOT_SESSION,
)

logger = logging.getLogger(__name__)

# Minimum text length to treat a post as meaningful
_MIN_TEXT_LEN = 20

# File extensions → media type
_PHOTO_EXTS   = {'.jpg', '.jpeg', '.png', '.webp'}
_VIDEO_EXTS   = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.gif'}


def _detect_media_type(path: str) -> str:
    """Return 'photo', 'video', or 'document' based on file extension."""
    ext = Path(path).suffix.lower()
    if ext in _PHOTO_EXTS:
        return 'photo'
    if ext in _VIDEO_EXTS:
        return 'video'
    return 'document'


# ---------------------------------------------------------------------------
# URL / username helpers
# ---------------------------------------------------------------------------

def _extract_username(raw: str) -> str | None:
    """
    Turn any of these into a bare channel slug:
      https://t.me/channel_name   →  channel_name
      @channel_name               →  channel_name
      channel_name                →  channel_name
    Returns None if the string doesn't look like a channel identifier.
    """
    raw = raw.strip()

    m = re.match(r"https?://t\.me/([a-zA-Z0-9_]{3,})/?$", raw)
    if m:
        return m.group(1).lower()

    m = re.match(r"@?([a-zA-Z0-9_]{3,})$", raw)
    if m:
        return m.group(1).lower()

    return None


# ---------------------------------------------------------------------------
# TelethonFetcher
# ---------------------------------------------------------------------------

class TelethonFetcher:
    """
    Wrapper around TelegramClient for fetching channel posts.

    The client is created lazily on first use and reused for the
    lifetime of the process. Call close() on application shutdown.
    """

    def __init__(self) -> None:
        # Cast api_id to int; keep empty string as 0 so is_available stays False
        self._api_id: int      = int(TELEGRAM_API_ID) if TELEGRAM_API_ID else 0
        self._api_hash: str    = TELEGRAM_API_HASH
        self._phone: str       = TELEGRAM_PHONE
        self._session_path: str = USERBOT_SESSION
        self._media_dir: Path   = Path(MEDIA_DIR)

        self._client: TelegramClient | None = None

    # ------------------------------------------------------------------ #
    # Public properties                                                    #
    # ------------------------------------------------------------------ #

    @property
    def is_available(self) -> bool:
        """True only when all three credentials are present in .env."""
        return bool(self._api_id and self._api_hash and self._phone)

    @property
    def session_exists(self) -> bool:
        """True when the saved session file is present on disk."""
        # Telethon appends '.session' if the name doesn't already end with it
        name = self._session_path
        if not name.endswith(".session"):
            name += ".session"
        return Path(name).exists()

    # ------------------------------------------------------------------ #
    # Client lifecycle                                                     #
    # ------------------------------------------------------------------ #

    async def _get_client(self) -> TelegramClient:
        """Return a connected TelegramClient (connects lazily on first call)."""
        if self._client is None:
            self._client = TelegramClient(
                self._session_path,
                self._api_id,
                self._api_hash,
            )
        if not self._client.is_connected():
            await self._client.connect()
        return self._client

    async def close(self) -> None:
        """Disconnect the client gracefully. Call once on application shutdown."""
        if self._client and self._client.is_connected():
            try:
                await self._client.disconnect()
                logger.info("Telethon client disconnected")
            except Exception as exc:
                logger.warning("Telethon disconnect failed (non-fatal): %s", exc)
        self._client = None

    # ------------------------------------------------------------------ #
    # Media download                                                       #
    # ------------------------------------------------------------------ #

    async def _save_media(self, client: TelegramClient, message) -> str | None:
        """
        Download media attached to `message` into MEDIA_DIR.
        Returns the absolute path as a string, or None on failure / no media.

        Strategy (api_id=2040 cannot access CDN DCs):
          1. Try to pick a non-CDN thumbnail directly from photo.sizes:
             prefer 'y' (1280px) → 'x' (800px) → 'm' (320px) → 's' (100px).
             PhotoSize objects with these type codes are stored on the main DC.
          2. As a last resort try full download (works for non-CDN files like stickers).
        """
        if not message.media:
            return None

        self._media_dir.mkdir(parents=True, exist_ok=True)
        dest = str(self._media_dir) + "/"

        # ── Attempt 1: pick a specific PhotoSize to avoid CDN ────────────────
        try:
            from telethon.tl.types import (  # noqa: PLC0415
                MessageMediaPhoto, MessageMediaDocument,
                PhotoSize, PhotoStrippedSize,
            )
            thumb_to_use = None

            if isinstance(message.media, MessageMediaPhoto):
                photo = message.media.photo
                if hasattr(photo, "sizes") and photo.sizes:
                    # Prefer larger non-stripped sizes, skip PhotoStrippedSize
                    for preferred in ("y", "x", "m", "s"):
                        match = next(
                            (s for s in photo.sizes
                             if isinstance(s, PhotoSize) and s.type == preferred),
                            None,
                        )
                        if match:
                            thumb_to_use = match
                            break

            if thumb_to_use is not None:
                path = await asyncio.wait_for(
                    client.download_media(message, file=dest, thumb=thumb_to_use),
                    timeout=25.0,
                )
                if path:
                    logger.info("Media saved (PhotoSize %s): %s", thumb_to_use.type, path)
                    return str(Path(path).resolve())
        except asyncio.TimeoutError:
            logger.warning("Media PhotoSize download timed out for msg %s", message.id)
        except Exception as exc:
            logger.warning("Media PhotoSize download failed for msg %s: %s", message.id, exc)

        # ── Attempt 2: full download (works for non-CDN files) ───────────────
        try:
            path = await asyncio.wait_for(
                client.download_media(message, file=dest),
                timeout=20.0,
            )
            if path:
                logger.info("Media saved (full download): %s", path)
                return str(Path(path).resolve())
        except asyncio.TimeoutError:
            logger.warning("Media full download timed out for msg %s", message.id)
        except Exception as exc:
            logger.warning("Media full download failed for msg %s: %s", message.id, exc)

        return None

    async def _save_media_via_http(
        self,
        channel_slug: str,
        msg_id: int,
        prefer_video: bool = False,
    ) -> str | None:
        """Backward-compat wrapper — returns first item from _save_all_media_via_http."""
        paths = await self._save_all_media_via_http(channel_slug, msg_id, prefer_video=prefer_video)
        return paths[0] if paths else None

    async def _save_all_media_via_http(
        self,
        channel_slug: str,
        msg_id: int,
        prefer_video: bool = False,
    ) -> list[str]:
        """
        HTTP CDN fallback: scrape t.me embed page and download ALL media via HTTPS.
        Returns list of downloaded file paths (may be empty on failure).
        Used when Telethon MTProto CDN (DC 203) is blocked by the hosting provider.
        Works for all public channels. Album posts may show multiple photos.
        """
        import re as _re
        import aiohttp as _aiohttp

        # Without &single=1 the embed may show the full album context
        embed_url = f"https://t.me/{channel_slug}/{msg_id}?embed=1"
        try:
            async with _aiohttp.ClientSession() as session:
                async with session.get(
                    embed_url,
                    timeout=_aiohttp.ClientTimeout(total=10),
                    headers={"User-Agent": "Mozilla/5.0"},
                ) as resp:
                    if resp.status != 200:
                        return []
                    html = await resp.text()
        except Exception as exc:
            logger.debug("HTTP CDN: embed fetch failed for %s/%s: %s", channel_slug, msg_id, exc)
            return []

        # Video: extract ALL <video src="..."> tags — reliable and specific
        video_pat = _re.compile(
            r'<video[^>]+src="(https://cdn[0-9]+\.telesco\.pe/file/[^"]+\.mp4[^"]*)"'
        )
        # Photo: strip channel avatar section first, then find ALL .jpg CDN URLs
        html_no_avatar = _re.sub(
            r'<i[^>]+class="[^"]*tgme_widget_message_user_photo[^"]*"[^>]*>.*?</i>',
            '', html, flags=_re.DOTALL
        )
        photo_pat = _re.compile(
            r"https://cdn[0-9]+\.telesco\.pe/file/[^\s\"'<>]+\.jpg(?:\?[^\s\"'<>]*)?"
        )

        # Collect all unique media URLs (videos first if preferred, then photos)
        collected: list[tuple[str, str]] = []  # (url, ext)
        seen_urls: set[str] = set()

        video_urls = video_pat.findall(html)
        photo_urls = [m.rstrip('"\'<> ') for m in photo_pat.findall(html_no_avatar)]

        if prefer_video:
            order = [(u, ".mp4") for u in video_urls] + [(u, ".jpg") for u in photo_urls]
        else:
            order = [(u, ".jpg") for u in photo_urls] + [(u, ".mp4") for u in video_urls]

        for url, ext in order:
            if url not in seen_urls:
                seen_urls.add(url)
                collected.append((url, ext))

        if not collected:
            logger.debug("HTTP CDN: no media URLs found in embed for %s/%s", channel_slug, msg_id)
            return []

        # Download all found media files
        self._media_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []
        for idx, (media_url, ext) in enumerate(collected):
            suffix = f"_{idx}" if idx > 0 else ""
            dest_path = self._media_dir / f"{channel_slug}_{msg_id}{suffix}{ext}"
            try:
                async with _aiohttp.ClientSession() as session:
                    async with session.get(
                        media_url,
                        timeout=_aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        if resp.status != 200:
                            logger.warning("HTTP CDN: download returned %s for %s/%s[%d]",
                                           resp.status, channel_slug, msg_id, idx)
                            continue
                        data = await resp.read()
                dest_path.write_bytes(data)
                saved_paths.append(str(dest_path))
                logger.info("HTTP CDN fallback saved [%d/%d] (%s): %s",
                            idx + 1, len(collected), ext, dest_path)
            except Exception as exc:
                logger.warning("HTTP CDN: download failed for %s/%s[%d]: %s",
                               channel_slug, msg_id, idx, exc)

        return saved_paths

    async def download_media_from_url(
        self,
        url: str,
    ) -> tuple[str | None, str | None]:
        """
        Download media from a specific Telegram post URL (t.me/channel/msg_id).

        Returns (file_path, media_type) on success, (None, None) otherwise.
        media_type is one of: 'photo', 'video', 'document'.

        Does NOT use forward/copy — fetches the raw message object and downloads
        the attached media directly, so the file has no trace of the source channel.
        """
        m = re.match(r'https?://t\.me/([A-Za-z0-9_]+)/(\d+)', url)
        if not m:
            logger.debug("download_media_from_url: not a t.me post URL — %s", url)
            return None, None

        channel_name = m.group(1)
        message_id   = int(m.group(2))

        if not self.is_available:
            logger.warning("download_media_from_url: Telethon credentials not set")
            return None, None

        if not self.session_exists:
            logger.warning("download_media_from_url: session file missing")
            return None, None

        try:
            client = await self._get_client()

            # Fetch the single message by its ID — no forward, no copy
            message = await client.get_messages(channel_name, ids=message_id)

            if not message:
                logger.debug("download_media_from_url: message %s/%s not found", channel_name, message_id)
                return None, None

            if not message.media:
                logger.debug("download_media_from_url: message %s/%s has no media", channel_name, message_id)
                return None, None

            path = await self._save_media(client, message)
            if not path:
                return None, None

            media_type = _detect_media_type(path)
            logger.info(
                "download_media_from_url: saved %s (%s) from %s",
                path, media_type, url,
            )
            return path, media_type

        except Exception as exc:
            logger.warning("download_media_from_url(%s) failed: %s", url, exc)
            return None, None

    # ------------------------------------------------------------------ #
    # Main fetch method                                                    #
    # ------------------------------------------------------------------ #

    async def fetch_channel_posts(
        self,
        channel_urls: list[str],
        limit: int = 10,
    ) -> list[dict]:
        """
        Fetch up to `limit` recent messages from each channel in `channel_urls`.

        Returns a combined list of post dicts (see module docstring for schema).
        Channels that are private, non-existent, or cause errors are skipped.
        """
        # ── Pre-flight checks ───────────────────────────────────────────────
        if not self.is_available:
            logger.warning(
                "Telethon: credentials missing in .env "
                "(TELEGRAM_API_ID / TELEGRAM_API_HASH / TELEGRAM_PHONE). "
                "Skipping Telethon fetch."
            )
            return []

        if not self.session_exists:
            logger.error(
                "Telethon: session file '%s' not found. "
                "Run   python auth_userbot.py   once to authorise, "
                "then restart the bot.",
                self._session_path,
            )
            return []

        client = await self._get_client()

        if not await client.is_user_authorized():
            logger.error(
                "Telethon: session exists but user is NOT authorised. "
                "Run   python auth_userbot.py   to re-authorise."
            )
            return []

        # ── Fetch posts ─────────────────────────────────────────────────────
        all_posts: list[dict] = []

        for raw_url in channel_urls:
            username = _extract_username(raw_url)
            if not username:
                logger.debug("Telethon: cannot parse username from '%s' — skipped", raw_url)
                continue

            # Resolve the entity (channel object)
            try:
                entity = await client.get_entity(username)
            except (UsernameInvalidError, UsernameNotOccupiedError, ValueError) as exc:
                logger.warning("Telethon: @%s not found — %s", username, exc)
                continue
            except ChannelPrivateError:
                logger.warning("Telethon: @%s is private — skipped", username)
                continue
            except FloodWaitError as exc:
                logger.warning(
                    "Telethon: flood-wait %ds triggered — skipping @%s",
                    exc.seconds, username,
                )
                continue
            except Exception as exc:
                logger.warning("Telethon: get_entity(@%s) failed — %s", username, exc)
                continue

            # Fetch messages
            try:
                messages = await client.get_messages(entity, limit=limit)
            except Exception as exc:
                logger.warning("Telethon: get_messages(@%s) failed — %s", username, exc)
                continue

            channel_slug = username.lower()
            channel_posts: list[dict] = []

            # ── Group messages by grouped_id for album support ───────────────
            # Telegram albums = multiple messages sharing the same grouped_id.
            # We want to collect ALL media from an album into one post entry.
            from collections import defaultdict as _defaultdict  # noqa: PLC0415
            from telethon.tl.types import MessageMediaDocument as _MMDoc  # noqa: PLC0415

            _albums: dict = _defaultdict(list)
            _standalone: list = []
            for msg in messages:
                if msg.grouped_id:
                    _albums[msg.grouped_id].append(msg)
                else:
                    _standalone.append(msg)

            # Build unified processing list: (representative_msg, all_msgs_in_group)
            _to_process: list[tuple] = []
            for _gid, _group in _albums.items():
                _group.sort(key=lambda m: m.id)  # ascending by ID
                _to_process.append((_group[0], _group))  # first msg = representative
            for msg in _standalone:
                _to_process.append((msg, [msg]))

            async def _download_one_media(msg) -> str | None:
                """Download media for a single message with Telethon + HTTP fallback."""
                if not msg.media:
                    return None
                path: str | None = None
                try:
                    path = await asyncio.wait_for(
                        self._save_media(client, msg),
                        timeout=15.0,
                    )
                except Exception:
                    pass
                if not path:
                    _prefer_vid = isinstance(getattr(msg, "media", None), _MMDoc)
                    try:
                        path = await asyncio.wait_for(
                            self._save_media_via_http(channel_slug, msg.id, prefer_video=_prefer_vid),
                            timeout=70.0,
                        )
                    except Exception:
                        pass
                return path

            for representative, group_msgs in _to_process:
                # Text comes from first message in group that has text
                text = ""
                for m in group_msgs:
                    t = (m.text or "").strip()
                    if t:
                        text = t
                        break

                has_any_media = any(bool(m.media) for m in group_msgs)

                # Skip service messages with no content
                if not text and not has_any_media:
                    continue
                if len(text) < _MIN_TEXT_LEN and not has_any_media:
                    continue

                display_text = text if text else "[media without text]"

                # Download ALL media from all messages in the group
                media_paths_fetched: list[str] = []

                if len(group_msgs) == 1:
                    # Single message — try HTTP all-media method first for potential album
                    msg = group_msgs[0]
                    if msg.media:
                        # Try to get all media from embed page (handles CDN block)
                        _prefer_vid = isinstance(getattr(msg, "media", None), _MMDoc)
                        try:
                            all_paths = await asyncio.wait_for(
                                self._save_all_media_via_http(channel_slug, msg.id, prefer_video=_prefer_vid),
                                timeout=90.0,
                            )
                            media_paths_fetched = all_paths
                        except Exception:
                            pass
                        # If HTTP found nothing, try Telethon
                        if not media_paths_fetched:
                            p = await _download_one_media(msg)
                            if p:
                                media_paths_fetched = [p]
                else:
                    # Album: download each message's media individually
                    for m in group_msgs:
                        p = await _download_one_media(m)
                        if p:
                            media_paths_fetched.append(p)

                link = f"https://t.me/{channel_slug}/{representative.id}"
                date: datetime = representative.date   # always timezone-aware (UTC)

                channel_posts.append({
                    # ── Telethon-native fields ──────────────────────────────
                    "channel":    channel_slug,
                    "message_id": representative.id,
                    "text":       display_text,
                    "date":       date,
                    "media_path": media_paths_fetched[0] if media_paths_fetched else None,
                    "media_paths": media_paths_fetched,
                    "has_media":  bool(media_paths_fetched) or has_any_media,
                    "link":       link,
                    # ── Normalised aliases (expected by Researcher / crew) ──
                    # These mirror the keys returned by the HTTP fetcher so the
                    # downstream pipeline doesn't need to know which fetcher ran.
                    "source":     f"https://t.me/{channel_slug}",
                    "url":        link,
                })

            logger.info(
                "Telethon: @%s — %d posts collected", channel_slug, len(channel_posts)
            )
            all_posts.extend(channel_posts)

        logger.info(
            "Telethon: total %d posts from %d channels",
            len(all_posts),
            len(channel_urls),
        )
        return all_posts


# ---------------------------------------------------------------------------
# Module-level singleton — one client per process
# ---------------------------------------------------------------------------

_fetcher = TelethonFetcher()


# ---------------------------------------------------------------------------
# Public convenience API
# ---------------------------------------------------------------------------

async def fetch_channel_posts(
    channel_urls: list[str],
    limit: int = 10,
) -> list[dict]:
    """
    Fetch posts from `channel_urls` using the shared TelethonFetcher.
    See TelethonFetcher.fetch_channel_posts() for full documentation.
    """
    return await _fetcher.fetch_channel_posts(channel_urls, limit)


async def close_userbot() -> None:
    """
    Disconnect the shared Telethon client.
    Call this once when the application shuts down (see main.py).
    """
    await _fetcher.close()


async def download_media_from_url(
    url: str,
) -> tuple[str | None, str | None]:
    """
    Download media from a Telegram post URL using the shared TelethonFetcher.
    Returns (file_path, media_type) or (None, None).
    No forward/copy — clean upload, no source channel trace.
    """
    return await _fetcher.download_media_from_url(url)


def _bot_entity_to_telethon(e: dict):
    """Convert a Bot API entity dict to a Telethon MessageEntity object."""
    t = e.get("type", "")
    offset = e.get("offset", 0)
    length = e.get("length", 0)
    if t == "bold":
        return MessageEntityBold(offset=offset, length=length)
    if t == "italic":
        return MessageEntityItalic(offset=offset, length=length)
    if t == "underline":
        return MessageEntityUnderline(offset=offset, length=length)
    if t == "strikethrough":
        return MessageEntityStrike(offset=offset, length=length)
    if t == "code":
        return MessageEntityCode(offset=offset, length=length)
    if t == "pre":
        return MessageEntityPre(offset=offset, length=length, language=e.get("language", ""))
    if t == "text_link":
        return MessageEntityTextUrl(offset=offset, length=length, url=e.get("url", ""))
    if t == "custom_emoji":
        return MessageEntityCustomEmoji(
            offset=offset, length=length,
            document_id=int(e["custom_emoji_id"]),
        )
    if t == "spoiler":
        return MessageEntitySpoiler(offset=offset, length=length)
    if t == "blockquote":
        return MessageEntityBlockquote(offset=offset, length=length)
    return None


async def publish_to_channel(channel_id: str, text: str, entities: list[dict]) -> bool:
    """
    Publish a message to the target channel via MTProto userbot.

    Unlike the Bot API, MTProto supports custom/premium emoji in channel posts
    when the userbot account has Telegram Premium. Falls back gracefully if
    the userbot is unavailable.

    Returns True on success, False on failure.
    """
    if not _fetcher.is_available:
        logger.warning("publish_to_channel: userbot not available")
        return False
    try:
        client = await _fetcher._get_client()
        telethon_entities = [
            obj for e in entities
            if (obj := _bot_entity_to_telethon(e)) is not None
        ]
        await client.send_message(
            entity=channel_id,
            message=text,
            formatting_entities=telethon_entities or None,
            parse_mode=None,
        )
        logger.info("publish_to_channel: sent %d chars to %s", len(text), channel_id)
        return True
    except Exception as exc:
        logger.error("publish_to_channel: failed: %s", exc)
        return False
