"""
Parse my_true_voise.html (Telegram Desktop HTML export) and extract
the author's post texts to use as style examples in LLM prompts.
"""

from pathlib import Path

from bs4 import BeautifulSoup


def parse_style_examples(html_file: str) -> list[str]:
    """
    Return a list of non-empty post texts extracted from the export.
    Skips system messages and media-only messages.
    """
    content = Path(html_file).read_text(encoding="utf-8")
    soup = BeautifulSoup(content, "lxml")

    texts: list[str] = []
    for msg in soup.select(".message.default, .message.default.joined"):
        text_div = msg.select_one("div.text")
        if not text_div:
            continue

        # get_text preserves line breaks from <br> tags
        raw = text_div.get_text(separator="\n").strip()

        # Skip very short or empty entries
        if len(raw) < 60:
            continue

        texts.append(raw)

    return texts


def build_style_prompt(html_file: str, max_examples: int = 6) -> str:
    """
    Return a formatted string with style reference examples
    ready to be embedded into an LLM system prompt.
    Limits to `max_examples` to keep the prompt size reasonable.
    """
    examples = parse_style_examples(html_file)
    if not examples:
        return ""

    # Prefer longer, more representative posts
    examples.sort(key=len, reverse=True)
    sample = examples[:max_examples]

    header = (
        "Ниже — примеры настоящих постов автора. "
        "Пиши ТОЧНО в таком же стиле: живой, личный, с юмором, "
        "от первого лица, с разговорными оборотами и авторскими интонациями.\n\n"
    )
    body = "\n\n───────────────\n\n".join(sample)
    return header + body
