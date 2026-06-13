"""
Fetch recent posts from all configured sources.

Sources (all applied a FETCH_CUTOFF_HOURS recency filter):
  1. Telegram channels   — Telethon (primary) or HTTP web-preview (fallback)
  2. RSS feeds           — feedparser
  3. Reddit JSON         — public /new.json endpoint (no auth required)
  4. Web digest sites    — HTML scraping (The Rundown AI, TLDR AI, Ben's Bites)
  5. Product Hunt        — HTML scraping (AI topics page)

After collection call filter_ads() to remove promotional content.

Public API (called by scheduler.run_pipeline_job):
  fetch_all_sources(sources_file) -> list[dict]
  filter_ads(posts)               -> list[dict]

Each post dict is guaranteed to have:
  source      str           — origin URL
  text        str           — post body
  date        str|datetime  — publication timestamp
  url         str           — permalink
  media_path  str|None      — local path if media was downloaded (Telethon only)
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from config import (
    FETCH_CUTOFF_HOURS,
    GITHUB_TRENDING_TOPICS,
    HUGGINGFACE_DAILY_PAPERS_LIMIT,
    HUGGINGFACE_TRENDING_MODELS_LIMIT,
    MAX_CHANNELS_PER_RUN,
    MAX_SOURCE_POSTS_PER_CHANNEL,
    MEDIA_DIR,
    PRODUCT_HUNT_URL,
    REDDIT_SUBREDDITS,
    RSS_SOURCES,
    SOURCES_FILE,
    WEB_DIGEST_SOURCES,
)

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru,en;q=0.9",
}

_MIN_TEXT_LEN = 40          # chars — shorter posts are skipped at fetch time
_CUTOFF_HOURS = FETCH_CUTOFF_HOURS   # from config (default 48)


# ---------------------------------------------------------------------------
# Recency filter helper
# ---------------------------------------------------------------------------

def _is_recent(date_val, hours: int | None = None) -> bool:
    """
    Return True if date_val is within the last `hours` hours (default: _CUTOFF_HOURS).
    Accepts datetime objects (aware or naive) or ISO-8601 strings.
    Unknown / unparseable dates are treated as recent (to avoid silent drops).
    """
    if date_val is None:
        return True

    h = hours if hours is not None else _CUTOFF_HOURS
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=h)

    if isinstance(date_val, datetime):
        dt = date_val
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt >= cutoff

    if isinstance(date_val, str) and date_val:
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(date_val[:25], fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt >= cutoff
            except ValueError:
                continue
        # feedparser struct_time comes as tuple-like string — accept to avoid silent drop
        return True

    return True


# ---------------------------------------------------------------------------
# Hard ad / promo filter
# ---------------------------------------------------------------------------

# Russian-language spam keywords — kept in Russian because that's the language
# of the source posts this filter needs to match.
# Add your own patterns here (in whatever language your sources use).
_AD_PATTERNS = [
    r"вебинар",                    # webinar
    r"скидк[иа]",                  # discount
    r"переходи по ссылке в профиле",  # "click the link in bio"
    r"записывайся",                # sign up / enroll
    r"мой курс",                   # my course
    r"early\s*bird",
    r"марафон",                    # marathon (online challenge)
    r"консультаци[яи]",            # consultation
    r"платный доступ",             # paid access
    r"промокод",                   # promo code
    r"успей купить",               # limited-time buy
    r"регистрируйся",              # register now
    r"бесплатный вебинар",         # free webinar
]

_AD_RE = re.compile("|".join(_AD_PATTERNS), re.IGNORECASE)

# Minimum meaningful length when a post has neither media nor a URL
_MIN_TEXT_NO_MEDIA = 120


def filter_ads(posts: list[dict]) -> list[dict]:
    """
    Remove promotional / low-quality posts from the collected feed.

    Removed when ANY of the following is true:
    • Post text matches one of _AD_PATTERNS
    • Post has no media AND no external URL AND text < _MIN_TEXT_NO_MEDIA chars
    """
    clean: list[dict] = []
    removed = 0

    for post in posts:
        text = post.get("text", "")

        if _AD_RE.search(text):
            removed += 1
            continue

        has_media = bool(post.get("media_path") or post.get("has_media"))
        has_link = bool(
            re.search(r"https?://", text)
            or (post.get("url", "") != post.get("source", ""))
        )
        if not has_media and not has_link and len(text) < _MIN_TEXT_NO_MEDIA:
            removed += 1
            continue

        clean.append(post)

    logger.info("filter_ads: removed %d/%d posts", removed, len(posts))
    return clean


# ---------------------------------------------------------------------------
# Telegram channel sources loader
# ---------------------------------------------------------------------------

def load_sources(sources_file: str = SOURCES_FILE) -> list[str]:
    """
    Read the sources file and return a deduplicated list of t.me URLs.
    Lines that are blank or start with '#' are ignored.
    """
    text = Path(sources_file).read_text(encoding="utf-8")
    seen: set[str] = set()
    sources: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls = re.findall(r"https?://t\.me/[^\s()]+", line)
        for url in urls:
            url = url.rstrip("/").lower()
            if url not in seen:
                seen.add(url)
                sources.append(url)

    return sources


# ---------------------------------------------------------------------------
# HTTP fallback fetcher for Telegram channels
# ---------------------------------------------------------------------------

def _to_web_preview(tme_url: str) -> str | None:
    m = re.match(r"https?://t\.me/([a-z0-9_]{3,})$", tme_url, re.IGNORECASE)
    if not m:
        return None
    return f"https://t.me/s/{m.group(1)}"


async def _http_fetch_one(
    channel_url: str,
    limit: int = MAX_SOURCE_POSTS_PER_CHANNEL,
) -> list[dict]:
    web_url = _to_web_preview(channel_url)
    if not web_url:
        logger.debug("HTTP: skipping non-channel URL: %s", channel_url)
        return []

    try:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers=_HEADERS,
        ) as client:
            resp = await client.get(web_url)

        if resp.status_code != 200:
            logger.warning("HTTP %s for %s", resp.status_code, web_url)
            return []

    except Exception as exc:
        logger.warning("HTTP fetch failed for %s: %s", web_url, exc)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    posts: list[dict] = []

    for msg in soup.select(".tgme_widget_message"):
        text_el = msg.select_one(".tgme_widget_message_text")
        if not text_el:
            continue

        text = text_el.get_text(separator="\n").strip()
        if len(text) < _MIN_TEXT_LEN:
            continue

        date_el = msg.select_one(".tgme_widget_message_date time")
        link_el = msg.select_one("a.tgme_widget_message_date")
        date_str = date_el.get("datetime", "") if date_el else ""

        if not _is_recent(date_str):
            continue

        posts.append({
            "source":     channel_url,
            "text":       text,
            "date":       date_str,
            "url":        link_el.get("href", web_url) if link_el else web_url,
            "media_path": None,
            # Check for media in web preview
            "has_media": bool(msg.select_one(".tgme_widget_message_photo_wrap"))
                       or bool(msg.select_one(".tgme_widget_message_video")),
        })

        if len(posts) >= limit:
            break

    logger.info("HTTP: fetched %d posts from %s", len(posts), channel_url)
    return posts


async def _http_fetch_all(
    sources: list[str],
    limit: int = MAX_SOURCE_POSTS_PER_CHANNEL,
) -> list[dict]:
    """Fetch all Telegram channels sequentially via HTTP (polite 1.5 s delay)."""
    all_posts: list[dict] = []

    for i, url in enumerate(sources):
        posts = await _http_fetch_one(url, limit)
        all_posts.extend(posts)
        if i < len(sources) - 1:
            await asyncio.sleep(1.5)

    logger.info("HTTP: total %d posts fetched", len(all_posts))
    return all_posts


# ---------------------------------------------------------------------------
# RSS feed fetcher
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OG media fetcher — grabs og:image / og:video from article URLs
# ---------------------------------------------------------------------------

async def _fetch_og_media(
    article_url: str,
    client: httpx.AsyncClient,
) -> str | None:
    """
    Fetch og:image or og:video:url from an article page and save to MEDIA_DIR.
    Returns local file path on success, None on any failure.
    Used to attach media to RSS / WebDigest items that have no media by default.
    """
    import hashlib as _hashlib
    import mimetypes as _mimetypes

    try:
        resp = await client.get(article_url, timeout=8.0)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")

        media_url: str | None = None
        is_video = False

        # Prefer video, then image
        for prop in ("og:video:secure_url", "og:video:url", "og:video",
                     "og:image:secure_url", "og:image"):
            tag = soup.find("meta", property=prop)
            if not tag:
                tag = soup.find("meta", attrs={"name": prop})
            if tag and tag.get("content", "").startswith("http"):
                media_url = tag["content"].strip()
                is_video = "video" in prop
                break

        if not media_url:
            return None

        # Don't download SVGs, data URIs, or tracking pixels
        low = media_url.lower()
        if any(x in low for x in (".svg", "data:", "1x1", "pixel", "tracking")):
            return None

        # Download the media
        media_resp = await client.get(media_url, timeout=15.0)
        if media_resp.status_code != 200 or len(media_resp.content) < 2000:
            return None  # skip tiny / broken files

        # Determine extension
        ct = media_resp.headers.get("content-type", "")
        if "mp4" in ct or "video" in ct or is_video:
            ext = ".mp4"
        elif "webp" in ct:
            ext = ".webp"
        elif "png" in ct:
            ext = ".png"
        elif "gif" in ct:
            ext = ".gif"
        else:
            ext = ".jpg"

        # Save with a hash-based name to avoid duplicates
        slug = _hashlib.md5(media_url.encode()).hexdigest()[:16]
        dest = Path(MEDIA_DIR) / f"web_{slug}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(media_resp.content)
        logger.debug("OG media saved: %s (%d bytes)", dest, len(media_resp.content))
        return str(dest)

    except Exception as exc:
        logger.debug("OG media fetch failed for %s: %s", article_url, exc)
        return None


async def fetch_rss_sources(rss_urls: list[str] | None = None) -> list[dict]:
    """
    Fetch posts from RSS feeds using feedparser.
    Only entries published within FETCH_CUTOFF_HOURS are returned.
    """
    if rss_urls is None:
        rss_urls = RSS_SOURCES
    if not rss_urls:
        return []

    try:
        import feedparser  # noqa: PLC0415
    except ImportError:
        logger.warning("feedparser not installed — skipping RSS. Run: pip install feedparser")
        return []

    all_posts: list[dict] = []

    for feed_url in rss_urls:
        try:
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(None, feedparser.parse, feed_url)
        except Exception as exc:
            logger.warning("RSS fetch failed for %s: %s", feed_url, exc)
            continue

        count = 0
        for entry in feed.get("entries", []):
            published = entry.get("published") or entry.get("updated") or ""
            parsed_time = entry.get("published_parsed") or entry.get("updated_parsed")
            if parsed_time:
                try:
                    import calendar  # noqa: PLC0415
                    ts = calendar.timegm(parsed_time)
                    published = datetime.fromtimestamp(ts, tz=timezone.utc)
                except Exception:
                    pass

            if not _is_recent(published):
                continue

            title = entry.get("title", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            if summary:
                summary = BeautifulSoup(summary, "lxml").get_text(separator=" ").strip()
            text = f"{title}\n\n{summary}".strip() if summary else title

            if len(text) < _MIN_TEXT_LEN:
                continue

            link = entry.get("link", feed_url)
            all_posts.append({
                "source":     feed_url,
                "text":       text[:1500],
                "date":       published,
                "url":        link,
                "media_path": None,
                "has_media":  False,
            })
            count += 1

        logger.info("RSS: %d recent entries from %s", count, feed_url)

    # Enrich RSS posts with og:image/og:video in parallel (best-effort)
    if all_posts:
        async with httpx.AsyncClient(
            timeout=20.0, follow_redirects=True, headers=_HEADERS,
        ) as og_client:
            tasks = [
                _fetch_og_media(p["url"], og_client)
                for p in all_posts
                if p.get("url") and p["url"].startswith("http")
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            idx = 0
            for p in all_posts:
                if p.get("url") and p["url"].startswith("http"):
                    r = results[idx]
                    idx += 1
                    if isinstance(r, str):
                        p["media_path"] = r
                        p["has_media"] = True
        logger.info("RSS: og:media enriched %d/%d posts",
                    sum(1 for p in all_posts if p.get("media_path")), len(all_posts))

    logger.info("RSS: total %d posts", len(all_posts))
    return all_posts


# ---------------------------------------------------------------------------
# Reddit JSON fetcher
# ---------------------------------------------------------------------------

async def fetch_reddit_json(subreddits: list[str] | None = None) -> list[dict]:
    """
    Fetch new posts from Reddit via the public /new.json endpoint.
    No authentication required. Only posts within FETCH_CUTOFF_HOURS.
    """
    if subreddits is None:
        subreddits = REDDIT_SUBREDDITS
    if not subreddits:
        return []

    all_posts: list[dict] = []
    reddit_headers = {
        **_HEADERS,
        "User-Agent": "ai-telegram-digest-bot/1.0 (educational project)",
    }

    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=True,
        headers=reddit_headers,
    ) as client:
        for sub in subreddits:
            url = f"https://www.reddit.com/r/{sub}/new.json?limit=25"
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    logger.warning("Reddit: HTTP %s for r/%s", resp.status_code, sub)
                    continue

                data = resp.json()
                posts_data = data.get("data", {}).get("children", [])
                count = 0

                for child in posts_data:
                    p = child.get("data", {})

                    if p.get("removed_by_category") or p.get("over_18"):
                        continue

                    created_utc = p.get("created_utc")
                    if created_utc:
                        post_dt = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
                    else:
                        post_dt = None

                    if not _is_recent(post_dt):
                        continue

                    title = p.get("title", "")
                    selftext = p.get("selftext", "") or ""
                    text = f"{title}\n\n{selftext}".strip() if selftext else title

                    if len(text) < _MIN_TEXT_LEN:
                        continue

                    permalink = "https://www.reddit.com" + p.get("permalink", "")
                    all_posts.append({
                        "source":     f"https://www.reddit.com/r/{sub}",
                        "text":       text[:1500],
                        "date":       post_dt or "",
                        "url":        permalink,
                        "media_path": None,
                        "has_media":  bool(p.get("url_overridden_by_dest") or p.get("thumbnail")),
                    })
                    count += 1

                logger.info("Reddit: %d recent posts from r/%s", count, sub)

            except Exception as exc:
                logger.warning("Reddit fetch failed for r/%s: %s", sub, exc)
            finally:
                await asyncio.sleep(1.0)

    logger.info("Reddit: total %d posts", len(all_posts))
    return all_posts


# ---------------------------------------------------------------------------
# Web digest scrapers (The Rundown AI, TLDR AI, Ben's Bites)
# ---------------------------------------------------------------------------

def _scrape_digest_page(source_url: str, html: str) -> list[dict]:
    """
    Extract article-level items from a newsletter/digest HTML page.
    Uses multiple CSS selector strategies to cover different site structures.
    """
    soup = BeautifulSoup(html, "lxml")
    posts: list[dict] = []
    now = datetime.now(tz=timezone.utc)

    # Strategy 1: look for semantic <article> elements or common card classes
    card_selectors = [
        "article",
        ".post-card", ".article-card", ".story-card", ".item-card",
        "[class*='post-item']", "[class*='article-item']", "[class*='story-item']",
        "[class*='newsletter-item']", "[class*='digest-item']",
    ]
    candidates: list = []
    for sel in card_selectors:
        candidates = soup.select(sel)
        if candidates:
            break

    for el in candidates[:25]:
        heading = el.select_one("h1, h2, h3, h4")
        if not heading:
            continue
        title = heading.get_text(separator=" ").strip()
        if len(title) < 15:
            continue

        link_el = heading.find("a") or el.select_one("a[href]")
        body_el = el.select_one(
            "p, [class*='summary'], [class*='description'], [class*='excerpt'], [class*='body']"
        )
        summary = body_el.get_text(separator=" ").strip() if body_el else ""

        text = f"{title}\n\n{summary}".strip() if summary else title
        if len(text) < _MIN_TEXT_LEN:
            continue

        href = link_el.get("href", "") if link_el else ""
        if href and not href.startswith("http"):
            href = urljoin(source_url, href)

        posts.append({
            "source":     source_url,
            "text":       text[:1500],
            "date":       now,
            "url":        href or source_url,
            "media_path": None,
        })

    # Strategy 2 fallback — heading + sibling paragraph pairs
    if not posts:
        for heading in soup.select("h2, h3")[:20]:
            title = heading.get_text().strip()
            if len(title) < 20:
                continue
            link_el = heading.find("a")
            href = link_el.get("href", "") if link_el else ""
            if href and not href.startswith("http"):
                href = urljoin(source_url, href)

            # Try next sibling paragraph
            next_sib = heading.find_next_sibling("p")
            summary = next_sib.get_text(separator=" ").strip() if next_sib else ""
            text = f"{title}\n\n{summary}".strip() if summary else title

            if len(text) < _MIN_TEXT_LEN:
                continue

            posts.append({
                "source":     source_url,
                "text":       text[:1500],
                "date":       now,
                "url":        href or source_url,
                "media_path": None,
            })

    return posts


async def fetch_web_digests(urls: list[str] | None = None) -> list[dict]:
    """
    Scrape newsletter/digest sites (The Rundown AI, TLDR AI, Ben's Bites).
    Best-effort: gracefully returns empty list on any failure.
    """
    if urls is None:
        urls = WEB_DIGEST_SOURCES
    if not urls:
        return []

    all_posts: list[dict] = []

    async with httpx.AsyncClient(
        timeout=25.0,
        follow_redirects=True,
        headers=_HEADERS,
    ) as client:
        for url in urls:
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    logger.warning("WebDigest: HTTP %s for %s", resp.status_code, url)
                    await asyncio.sleep(1.0)
                    continue
                posts = _scrape_digest_page(url, resp.text)
                all_posts.extend(posts)
                logger.info("WebDigest: %d items from %s", len(posts), url)
            except Exception as exc:
                logger.warning("WebDigest scrape failed for %s: %s", url, exc)
            await asyncio.sleep(1.5)

    logger.info("WebDigest: total %d items", len(all_posts))
    return all_posts


# ---------------------------------------------------------------------------
# Product Hunt AI topics scraper
# ---------------------------------------------------------------------------

async def fetch_product_hunt(url: str | None = None) -> list[dict]:
    """
    Scrape Product Hunt AI topics page for new AI tools.
    Best-effort: returns empty list if the page is JS-rendered or unreachable.
    Results go into the 'Полезные сервисы' category for the analyst.
    """
    if url is None:
        url = PRODUCT_HUNT_URL
    if not url:
        return []

    ph_headers = {
        **_HEADERS,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        # Product Hunt requires a browser-like Accept header
    }

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers=ph_headers,
        ) as client:
            resp = await client.get(url)

        if resp.status_code != 200:
            logger.warning("ProductHunt: HTTP %s", resp.status_code)
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        posts: list[dict] = []
        now = datetime.now(tz=timezone.utc)

        # PH renders with React but some product data survives in the initial HTML.
        # Selectors covering several known PH HTML patterns:
        card_selectors = [
            "[data-test='product-item']",
            "li[class*='styles_item']",
            "li[class*='product']",
            "div[class*='ProductItem']",
        ]
        cards: list = []
        for sel in card_selectors:
            cards = soup.select(sel)
            if cards:
                break

        for card in cards[:20]:
            name_el = card.select_one(
                "h3, h2, [class*='name'], [class*='title'], [class*='Name'], [class*='Title']"
            )
            desc_el = card.select_one(
                "p, [class*='description'], [class*='tagline'], [class*='Description']"
            )
            link_el = card.select_one("a[href]")

            if not name_el:
                continue

            name = name_el.get_text().strip()
            desc = desc_el.get_text().strip() if desc_el else ""
            text = f"{name}\n\n{desc}".strip() if desc else name

            if len(text) < _MIN_TEXT_LEN:
                continue

            href = link_el.get("href", "") if link_el else ""
            if href and not href.startswith("http"):
                href = f"https://www.producthunt.com{href}"

            posts.append({
                "source":     url,
                "text":       text[:1000],
                "date":       now,
                "url":        href or url,
                "media_path": None,
            })

        logger.info("ProductHunt: %d items fetched", len(posts))
        return posts

    except Exception as exc:
        logger.warning("ProductHunt scrape failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Public API — called by scheduler.run_pipeline_job()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# HuggingFace: daily papers + trending models
# ---------------------------------------------------------------------------

async def fetch_huggingface(
    papers_limit: int = HUGGINGFACE_DAILY_PAPERS_LIMIT,
    models_limit: int = HUGGINGFACE_TRENDING_MODELS_LIMIT,
) -> list[dict]:
    """Fetch HuggingFace daily papers and trending models via their public API."""
    now = datetime.now(timezone.utc)
    posts: list[dict] = []

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=_HEADERS) as client:
        # ── Daily papers ─────────────────────────────────────────────────
        try:
            resp = await client.get(
                f"https://huggingface.co/api/daily_papers?limit={papers_limit}"
            )
            if resp.status_code == 200:
                for item in resp.json():
                    paper = item.get("paper", {})
                    title = paper.get("title", "").strip()
                    abstract = paper.get("summary", "").strip()
                    upvotes = paper.get("upvotes", 0)
                    paper_id = paper.get("id", "")
                    if not title or upvotes < 3:
                        continue
                    thumbnail = paper.get("thumbnailUrl", "") or ""
                    # Download thumbnail if available
                    media_path = None
                    if thumbnail.startswith("http"):
                        try:
                            img_resp = await client.get(thumbnail, timeout=8.0)
                            if img_resp.status_code == 200 and len(img_resp.content) > 2000:
                                import hashlib as _h
                                slug = _h.md5(thumbnail.encode()).hexdigest()[:12]
                                dest = Path(MEDIA_DIR) / f"hf_paper_{slug}.jpg"
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                dest.write_bytes(img_resp.content)
                                media_path = str(dest)
                        except Exception:
                            pass
                    text = f"{title}\n\n{abstract[:600]}" if abstract else title
                    posts.append({
                        "source":     "https://huggingface.co/papers",
                        "text":       text[:1500],
                        "date":       now,
                        "url":        f"https://huggingface.co/papers/{paper_id}",
                        "media_path": media_path,
                        "has_media":  bool(media_path),
                    })
        except Exception as exc:
            logger.warning("HuggingFace daily papers failed: %s", exc)

        # ── Trending models ───────────────────────────────────────────────
        try:
            resp = await client.get(
                f"https://huggingface.co/api/models?sort=trendingScore&direction=-1"
                f"&limit={models_limit}&full=false"
            )
            if resp.status_code == 200:
                for m in resp.json():
                    model_id = m.get("id", "")
                    pipeline = m.get("pipeline_tag", "")
                    downloads = m.get("downloads", 0)
                    if not model_id or downloads < 1000:
                        continue
                    # Skip boring/NSFW model names
                    low = model_id.lower()
                    if any(x in low for x in ("uncensored", "nsfw", "nude", "xxx")):
                        continue
                    text = (
                        f"🤗 Trending: {model_id}\n"
                        f"Тип: {pipeline or 'не указан'}\n"
                        f"Скачиваний: {downloads:,}"
                    )
                    posts.append({
                        "source":     "https://huggingface.co",
                        "text":       text,
                        "date":       now,
                        "url":        f"https://huggingface.co/{model_id}",
                        "media_path": None,
                        "has_media":  False,
                    })
        except Exception as exc:
            logger.warning("HuggingFace trending models failed: %s", exc)

    logger.info("HuggingFace: %d items fetched", len(posts))
    return posts


# ---------------------------------------------------------------------------
# GitHub: trending AI repos via Search API (no auth required)
# ---------------------------------------------------------------------------

async def fetch_github_trending(topics: list[str] | None = None) -> list[dict]:
    """
    Fetch recently-starred AI repos from GitHub Search API.
    Groups by topic, deduplicates, returns up to 10 most relevant items.
    """
    if topics is None:
        topics = GITHUB_TRENDING_TOPICS

    from datetime import timedelta as _td
    cutoff = (datetime.now(timezone.utc) - _td(days=3)).strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc)
    seen: set[str] = set()
    posts: list[dict] = []

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=_HEADERS) as client:
        for topic in topics[:4]:  # limit to 4 topics to avoid rate limit
            try:
                resp = await client.get(
                    "https://api.github.com/search/repositories",
                    params={
                        "q": f"topic:{topic} created:>{cutoff}",
                        "sort": "stars",
                        "order": "desc",
                        "per_page": 5,
                    },
                    headers={
                        **_HEADERS,
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                    timeout=10.0,
                )
                if resp.status_code != 200:
                    continue

                for repo in resp.json().get("items", []):
                    full_name = repo.get("full_name", "")
                    if full_name in seen:
                        continue
                    stars = repo.get("stargazers_count", 0)
                    if stars < 50:  # skip tiny repos
                        continue
                    seen.add(full_name)
                    desc = (repo.get("description") or "").strip()
                    lang = repo.get("language") or ""
                    text = (
                        f"⭐ GitHub: {full_name} ({stars:,} stars)\n"
                        + (f"{desc}\n" if desc else "")
                        + (f"Язык: {lang}" if lang else "")
                    ).strip()
                    posts.append({
                        "source":     "https://github.com",
                        "text":       text[:800],
                        "date":       now,
                        "url":        repo.get("html_url", f"https://github.com/{full_name}"),
                        "media_path": None,
                        "has_media":  False,
                    })
                await asyncio.sleep(0.5)  # gentle rate limiting
            except Exception as exc:
                logger.warning("GitHub trending for topic=%s failed: %s", topic, exc)

    # Sort by stars desc, cap at 8
    posts.sort(key=lambda p: int(p["text"].split("stars)")[0].split("(")[-1].replace(",","")) if "stars)" in p["text"] else 0, reverse=True)
    posts = posts[:8]
    logger.info("GitHub: %d trending AI repos fetched", len(posts))
    return posts


async def fetch_all_sources(
    sources_file: str = SOURCES_FILE,
    max_channels: int = MAX_CHANNELS_PER_RUN,
) -> list[dict]:
    """
    Fetch posts from ALL configured sources within FETCH_CUTOFF_HOURS:
      • Telegram channels  (Telethon primary, HTTP fallback)
      • RSS feeds
      • Reddit /new.json
      • Web digest sites   (The Rundown AI, TLDR AI, Ben's Bites)
      • Product Hunt AI    (new tools)

    Returns a flat list of post dicts.
    Does NOT apply filter_ads() — call it separately after this function.
    """
    sources = load_sources(sources_file)[:max_channels]
    logger.info("Sources loaded: %d Telegram channels", len(sources))

    # ── Telegram: Telethon (primary) ────────────────────────────────────────
    tg_posts: list[dict] = []
    try:
        from parsers.telegram_userbot import _fetcher  # noqa: PLC0415
        from parsers.telegram_userbot import (  # noqa: PLC0415
            fetch_channel_posts as _tg_fetch,
        )

        if _fetcher.is_available:
            if not _fetcher.session_exists:
                logger.warning(
                    "Telethon: session missing — run python auth_userbot.py. "
                    "Falling back to HTTP."
                )
            else:
                logger.info("Telethon: fetching from %d channels …", len(sources))
                try:
                    raw = await _tg_fetch(sources, limit=MAX_SOURCE_POSTS_PER_CHANNEL)
                    tg_posts = [p for p in raw if _is_recent(p.get("date"))]
                    logger.info(
                        "Telethon: %d posts collected (%d within %dh window)",
                        len(raw), len(tg_posts), _CUTOFF_HOURS,
                    )
                except Exception as exc:
                    logger.error("Telethon fetch error: %s — falling back to HTTP", exc)
        else:
            logger.info("Telethon: credentials not configured — using HTTP web-preview.")

    except ImportError:
        logger.info("telethon not installed — using HTTP web-preview.")

    # ── Telegram: HTTP fallback ──────────────────────────────────────────────
    if not tg_posts:
        logger.info("HTTP fallback: fetching Telegram channels …")
        tg_posts = await _http_fetch_all(sources)

    # ── RSS feeds ────────────────────────────────────────────────────────────
    rss_posts = await fetch_rss_sources()

    # ── Reddit JSON ──────────────────────────────────────────────────────────
    reddit_posts = await fetch_reddit_json()

    # ── Web digest sites ─────────────────────────────────────────────────────
    web_posts = await fetch_web_digests()

    # ── Product Hunt ─────────────────────────────────────────────────────────
    ph_posts = await fetch_product_hunt()

    # ── HuggingFace: daily papers + trending models ──────────────────────────
    hf_posts = await fetch_huggingface()

    # ── GitHub: trending AI repos ────────────────────────────────────────────
    gh_posts = await fetch_github_trending()

    all_posts = tg_posts + rss_posts + reddit_posts + web_posts + ph_posts + hf_posts + gh_posts
    logger.info(
        "fetch_all_sources: Telegram=%d  RSS=%d  Reddit=%d  WebDigest=%d  ProductHunt=%d  HuggingFace=%d  GitHub=%d  TOTAL=%d",
        len(tg_posts), len(rss_posts), len(reddit_posts), len(web_posts), len(ph_posts), len(hf_posts), len(gh_posts), len(all_posts),
    )
    return all_posts


# ---------------------------------------------------------------------------
# Legacy alias
# ---------------------------------------------------------------------------

async def fetch_telegram_posts(
    channel_url: str,
    limit: int = MAX_SOURCE_POSTS_PER_CHANNEL,
) -> list[dict]:
    """Single-channel HTTP fetch. Kept for backward compatibility."""
    return await _http_fetch_one(channel_url, limit)
