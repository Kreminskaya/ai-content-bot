"""
ReviewsAgent — collects live user reviews for a digest topic.

Runs BEFORE WriterCrew when admin clicks a topic button.
Results are passed to Researcher as a "User reviews" context section.

Data sources (all public, no auth required):
  1. Reddit search — /search.json across key AI subreddits
  2. GitHub Issues  — basic fetch if topic URL points to github.com
  3. HuggingFace discussions — basic fetch if URL points to huggingface.co

Design principles:
  • Non-blocking: if nothing found → returns empty ReviewsResult,
    the main pipeline continues without user reviews.
  • No extra LLM call: raw collected text is formatted into a readable block
    and passed directly to Researcher. Researcher already has analysis skills.
  • Hard timeout: entire collection capped at 15 seconds to not delay the pipeline.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import quote_plus, urlparse

import httpx

logger = logging.getLogger(__name__)

# AI subreddits to search — combined in one request with + operator
_REVIEW_SUBREDDITS = (
    "LocalLLaMA+MachineLearning+StableDiffusion+comfyui+aivideo+singularity"
    "+Artificial+ChatGPT+OpenAI+stablediffusion_ui"
)

_REDDIT_HEADERS = {
    "User-Agent": "ai-telegram-review-collector/1.0 (educational project)",
    "Accept-Language": "en-US,en;q=0.9",
}

_MAX_REVIEWS      = 8    # max number of Reddit posts to include
_MAX_TEXT_PER_REV = 400  # chars per review (keep context tight)
_FETCH_TIMEOUT    = 12.0 # seconds per HTTP request
_TOTAL_TIMEOUT    = 15.0 # seconds for the entire collection


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class ReviewsResult:
    reviews: list[str] = field(default_factory=list)   # individual review texts
    sources: list[str] = field(default_factory=list)   # source URLs
    summary: str = ""   # formatted block passed to Researcher


# ---------------------------------------------------------------------------
# Keyword extractor
# ---------------------------------------------------------------------------

def _extract_keywords(title: str) -> str:
    """
    Extract the most meaningful search term from a digest topic title.
    Strips common filler words and returns 2–4 key tokens.

    Examples (RU titles are supported alongside EN titles):
      "Runway выпустил инструмент автоматического монтажа" → "Runway автомонтаж"
      "Gemini 2.5 Pro — лучший по бенчмаркам" → "Gemini 2.5 Pro"
      "ComfyUI новый нод для LoRA" → "ComfyUI LoRA"
    """
    # Russian filler words stripped when topic titles are in Russian
    STOP_RU = {
        "выпустил", "выпустила", "анонсировал", "анонсировала", "запустил",
        "представил", "обновил", "добавил", "теперь", "новый", "новая", "новое",
        "для", "по", "на", "от", "из", "или", "это", "лучший", "лучшая",
        "самый", "самая", "стал", "стала", "стало", "появился", "появилась",
        "что", "как", "уже", "вот", "если", "при", "за", "до", "после",
        "инструмент", "инструменты", "сервис", "платформа", "функция",
        "модель", "нейросеть",
    }
    tokens = re.split(r"[\s\-–—,:;/]+", title)
    keywords = [
        t for t in tokens
        if len(t) >= 3 and t.lower() not in STOP_RU
    ]
    # Take first 4 meaningful tokens
    return " ".join(keywords[:4])


# ---------------------------------------------------------------------------
# Reddit search
# ---------------------------------------------------------------------------

async def _search_reddit(query: str, client: httpx.AsyncClient) -> list[dict]:
    """
    Search across key AI subreddits via public Reddit JSON API.
    Returns list of dicts with keys: text, url, score, sub.
    """
    encoded = quote_plus(query)
    url = (
        f"https://www.reddit.com/r/{_REVIEW_SUBREDDITS}/search.json"
        f"?q={encoded}&restrict_sr=on&sort=relevance&t=month&limit=20"
    )
    try:
        resp = await client.get(url, timeout=_FETCH_TIMEOUT)
        if resp.status_code != 200:
            logger.debug("ReviewsAgent Reddit search HTTP %s for query '%s'", resp.status_code, query)
            return []

        children = resp.json().get("data", {}).get("children", [])
        results = []
        for child in children:
            p = child.get("data", {})
            if p.get("removed_by_category") or p.get("over_18") or p.get("stickied"):
                continue

            title    = p.get("title", "")
            selftext = (p.get("selftext", "") or "").strip()
            score    = p.get("score", 0)
            sub      = p.get("subreddit", "")
            permalink = "https://www.reddit.com" + p.get("permalink", "")

            # Skip posts with very low engagement (likely spam/noise)
            if score < 2:
                continue

            body = f"{title}\n{selftext}".strip() if selftext else title
            if len(body) < 20:
                continue

            results.append({
                "text":  body[:_MAX_TEXT_PER_REV],
                "url":   permalink,
                "score": score,
                "sub":   sub,
            })

        # Sort by score, take top N
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:_MAX_REVIEWS]

    except Exception as exc:
        logger.debug("ReviewsAgent Reddit search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# GitHub Issues (basic)
# ---------------------------------------------------------------------------

async def _fetch_github_issues(github_url: str, client: httpx.AsyncClient) -> list[dict]:
    """
    Fetch top issues from a GitHub repo using the public search API.
    Only works for github.com URLs (owner/repo pattern).
    Returns list of dicts with keys: text, url.
    """
    parsed = urlparse(github_url)
    # Extract owner/repo from path like /owner/repo or /owner/repo/...
    path_parts = [p for p in parsed.path.split("/") if p]
    if len(path_parts) < 2:
        return []

    owner, repo = path_parts[0], path_parts[1]
    repo_slug = f"{owner}/{repo}"

    url = (
        f"https://api.github.com/search/issues"
        f"?q=repo:{repo_slug}+type:issue&sort=comments&per_page=5"
    )
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        resp = await client.get(url, headers=headers, timeout=_FETCH_TIMEOUT)
        if resp.status_code != 200:
            logger.debug("ReviewsAgent GitHub issues HTTP %s for %s", resp.status_code, repo_slug)
            return []

        items = resp.json().get("items", [])
        results = []
        for issue in items[:5]:
            title = issue.get("title", "")
            body  = (issue.get("body", "") or "").strip()[:200]
            comments = issue.get("comments", 0)
            html_url = issue.get("html_url", "")
            state    = issue.get("state", "")

            label = "🐛" if "bug" in issue.get("title", "").lower() else "💬"
            text = f"{label} [{state}] {title}"
            if body:
                text += f"\n{body}"
            if comments:
                text += f" ({comments} comments)"

            results.append({"text": text, "url": html_url})

        return results

    except Exception as exc:
        logger.debug("ReviewsAgent GitHub issues failed for %s: %s", repo_slug, exc)
        return []


# ---------------------------------------------------------------------------
# HuggingFace (basic model card / community tab via scraping)
# ---------------------------------------------------------------------------

async def _fetch_hf_discussions(hf_url: str, client: httpx.AsyncClient) -> list[dict]:
    """
    Fetch the HuggingFace model/space page and extract the first few
    community discussion titles from the HTML.
    Very lightweight — just a scrape of the discussions tab.
    """
    # Normalise URL: /owner/repo → /owner/repo/discussions
    parsed = urlparse(hf_url)
    path = parsed.path.rstrip("/")
    discussions_url = f"https://huggingface.co{path}/discussions"

    try:
        resp = await client.get(
            discussions_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; bot/1.0)"},
            timeout=_FETCH_TIMEOUT,
        )
        if resp.status_code != 200:
            return []

        # Extract discussion titles from the page HTML
        # HF renders titles in <h3> or <a> tags with discussion content
        titles = re.findall(
            r'class="[^"]*discussion[^"]*"[^>]*>\s*<[^>]+>\s*([^<]{10,200})',
            resp.text,
        )
        if not titles:
            # Fallback: grab any reasonable anchor text that looks like discussion
            titles = re.findall(
                r'"discussionTitle"[^>]*>([^<]{10,200})<',
                resp.text,
            )

        results = []
        for title in titles[:5]:
            title = title.strip()
            if title:
                results.append({"text": f"💬 HF discussion: {title}", "url": discussions_url})

        return results

    except Exception as exc:
        logger.debug("ReviewsAgent HF discussions failed for %s: %s", hf_url, exc)
        return []


# ---------------------------------------------------------------------------
# Format results into a readable summary block
# ---------------------------------------------------------------------------

def _format_summary(
    reddit_results: list[dict],
    github_results: list[dict],
    hf_results: list[dict],
    keywords: str,
) -> ReviewsResult:
    """
    Combine collected reviews into a ReviewsResult.
    The summary field is a formatted text block ready for Researcher's context.
    """
    reviews: list[str] = []
    sources: list[str] = []

    if reddit_results:
        reviews.append(f"=== Reddit (поиск: «{keywords}») ===")
        for r in reddit_results:
            reviews.append(f"r/{r['sub']} | ⬆{r['score']}\n{r['text']}")
            sources.append(r["url"])

    if github_results:
        reviews.append("=== GitHub Issues ===")
        for r in github_results:
            reviews.append(r["text"])
            sources.append(r["url"])

    if hf_results:
        reviews.append("=== HuggingFace Discussions ===")
        for r in hf_results:
            reviews.append(r["text"])
            sources.append(r["url"])

    if not reviews:
        return ReviewsResult()

    summary = "\n\n".join(reviews)
    return ReviewsResult(
        reviews=reviews,
        sources=list(dict.fromkeys(sources)),  # deduplicate, preserve order
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def run_reviews(topic_title: str, topic_url: str) -> ReviewsResult:
    """
    Collect live user reviews for a digest topic.

    Searches Reddit (always) + GitHub (if topic_url is github.com)
    + HuggingFace (if topic_url is huggingface.co).

    Hard-capped at TOTAL_TIMEOUT seconds — never blocks the main pipeline.
    Returns empty ReviewsResult if nothing found or if all sources fail.
    """
    keywords = _extract_keywords(topic_title)
    if not keywords:
        keywords = topic_title[:50]

    logger.info(
        "ReviewsAgent: searching for '%s' (url=%s)",
        keywords, topic_url or "none",
    )

    parsed_url = urlparse(topic_url or "")
    hostname   = parsed_url.hostname or ""

    async def _collect() -> ReviewsResult:
        reddit_results: list[dict] = []
        github_results: list[dict] = []
        hf_results:     list[dict] = []

        async with httpx.AsyncClient(
            follow_redirects=True,
            headers=_REDDIT_HEADERS,
        ) as client:
            # Always search Reddit
            reddit_task = asyncio.create_task(_search_reddit(keywords, client))

            # Source-specific fetchers
            github_task = None
            hf_task     = None

            if "github.com" in hostname:
                github_task = asyncio.create_task(_fetch_github_issues(topic_url, client))
            if "huggingface.co" in hostname:
                hf_task = asyncio.create_task(_fetch_hf_discussions(topic_url, client))

            reddit_results = await reddit_task
            if github_task:
                github_results = await github_task
            if hf_task:
                hf_results = await hf_task

        return _format_summary(reddit_results, github_results, hf_results, keywords)

    try:
        result = await asyncio.wait_for(_collect(), timeout=_TOTAL_TIMEOUT)
        total = len(result.reviews)
        logger.info(
            "ReviewsAgent: found %d entries for '%s'",
            total, keywords,
        )
        return result
    except asyncio.TimeoutError:
        logger.warning("ReviewsAgent: timed out after %.0fs for '%s'", _TOTAL_TIMEOUT, keywords)
        return ReviewsResult()
    except Exception as exc:
        logger.warning("ReviewsAgent: unexpected error for '%s': %s", keywords, exc)
        return ReviewsResult()
