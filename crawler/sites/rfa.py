"""
RFA Burmese -- https://www.rfa.org/burmese

Stack: Arc XP (Fusion). Every article page embeds a `Fusion.globalContent`
JSON object holding the complete article model -- body blocks, taxonomy,
credits, dates -- so parsing is JSON rather than HTML scraping. A JSON-LD
NewsArticle block is also present and is used as a fallback.

robots.txt: articles are allowed. Only /burmese/search/?query=* and any path
ending in `story_archive` are disallowed -- the latter would have been a
convenient archive route, so we deliberately do not touch it. No Crawl-delay is
published; we use 2s.

Discovery is the cleanest of any outlet so far. Arc exposes DATE-ADDRESSABLE
daily sitemaps:

    /arc/outboundfeeds/burmese/sitemap/<YYYY-MM-DD>/?outputType=xml

The published index lists only the last 365 days, but arbitrary earlier dates
resolve fine. So we simply enumerate every day in the requested range -- one
request per day, ~633 for Jan 2025 onward, exact and complete. No pagination
guessing, no over-fetching of out-of-range articles, no external archive needed.

Note `sitemap/`, `sitemap2/` and `sitemap3/` are aliases returning identical
content; only `sitemap/` is used.

URL FORMAT CHANGE: RFA switched article URLs partway through the range --
legacy `/burmese/news/<slug>-MMDDYYYYHHMMSS.html` up to ~mid-2025, and
`/burmese/news/YYYY/MM/DD/<slug>/` from 2026. Both appear in the sitemaps and
both must be matched; see _is_article().

IMPORTANT -- publishing gap: RFA Burmese output declines sharply through early
2025 (~17/day in January, ~6/day by April) and the daily sitemaps return a
valid but EMPTY urlset for roughly June-December 2025. Those dates appear in
RFA's own 365-day index, so the emptiness is what RFA publishes, not a hole in
our crawl. Expect a real gap in the corpus for that period and say so in any
analysis rather than treating it as missing data.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Iterator

from bs4 import BeautifulSoup

from ..core import Context, record

name = "rfa"
outlet = "rfa_burmese"
base_url = "https://www.rfa.org"
delay = 2.0          # no published Crawl-delay

DAY_SITEMAP = (
    "https://rfa.org/arc/outboundfeeds/burmese/sitemap/{day}/?outputType=xml"
)
LOC_RE = re.compile(r"<loc>\s*(https?://(?:www\.)?rfa\.org/burmese/[^<\s]+)\s*</loc>")

# RFA changed its article URL format partway through the corpus range, and both
# forms appear in the daily sitemaps:
#
#   legacy  (to ~mid-2025)  /burmese/news/<slug>-MMDDYYYYHHMMSS.html
#   current (2026 onward)   /burmese/news/YYYY/MM/DD/<slug>/
#
# Matching only the legacy `.html` form silently drops every 2026 article while
# discovery still reports success, so both patterns are required. Anchoring on
# these two shapes (rather than accepting any /burmese/ URL) keeps section
# landing pages and tag indexes out of the queue.
LEGACY_ARTICLE_RE = re.compile(r"-\d{14}\.html$")
DATED_ARTICLE_RE = re.compile(r"/\d{4}/\d{2}/\d{2}/[^/]+/?$")


def _is_article(url: str) -> bool:
    path = url.split("?")[0]
    return bool(LEGACY_ARTICLE_RE.search(path) or DATED_ARTICLE_RE.search(path))


FUSION_KEY = "Fusion.globalContent="


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def _days(start: str, end: str | None) -> Iterator[str]:
    d = date.fromisoformat(start)
    last = date.fromisoformat(end) if end else datetime.now(timezone.utc).date()
    while d <= last:
        yield d.isoformat()
        d += timedelta(days=1)


def discover(ctx: Context) -> Iterator[str]:
    # Chronological with a single cursor: the day order is deterministic, so
    # resuming means "continue after the last day we finished".
    cursor = ctx.state.get("day_cursor")
    empty_days = ctx.state.get("empty_days", 0)
    seen_days = 0

    for day in _days(ctx.start, ctx.end):
        if cursor and day <= cursor:
            continue
        if ctx.max_pages and seen_days >= ctx.max_pages:
            ctx.log.info("rfa: hit --max-pages %d days", ctx.max_pages)
            break

        status, body, err = ctx.fetch(DAY_SITEMAP.format(day=day))
        seen_days += 1

        if status != 200 or not body:
            # Do not advance the cursor: a transient failure should be retried
            # on the next run rather than silently skipping a day.
            ctx.log.warning("rfa: %s sitemap failed (%s), will retry next run",
                            day, err or status)
            continue

        urls = []
        for u in LOC_RE.findall(body.decode("utf-8", "replace")):
            u = u.split("?")[0]
            if _is_article(u):
                urls.append(u.replace("https://rfa.org/", f"{base_url}/"))

        if urls:
            for u in dict.fromkeys(urls):
                yield u
            ctx.log.info("rfa: %s -> %d articles", day, len(urls))
        else:
            empty_days += 1
            ctx.state.set("empty_days", empty_days)

        ctx.state.set("day_cursor", day)

    if empty_days:
        ctx.log.info("rfa: %d day(s) in range published nothing "
                     "(expected -- see the 2025 gap note in this module)",
                     empty_days)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _extract_fusion(text: str) -> dict | None:
    """Pull Fusion.globalContent out of the page by brace matching.

    A regex cannot do this safely: the object contains braces inside string
    literals, so we walk it while tracking string state.
    """
    i = text.find(FUSION_KEY)
    if i == -1:
        return None
    i += len(FUSION_KEY)
    if i >= len(text) or text[i] != "{":
        return None

    depth = 0
    in_str = False
    esc = False
    j = i
    while j < len(text):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    j += 1
                    break
        j += 1

    try:
        return json.loads(text[i:j])
    except json.JSONDecodeError:
        return None


def _html_text(fragment: str) -> str:
    """Strip inline markup from a content_element, keeping the text as-is."""
    if "<" not in fragment:
        return fragment
    return BeautifulSoup(fragment, "lxml").get_text("", strip=False)


def _iter_jsonld(soup: BeautifulSoup):
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for obj in data if isinstance(data, list) else [data]:
            if isinstance(obj, dict):
                yield obj


def parse(html: bytes, url: str) -> dict | None:
    text = html.decode("utf-8", "replace")
    gc = _extract_fusion(text)

    if gc is not None and gc.get("subtype") in (None, "story", "gallery", "video"):
        # Body: paragraph blocks only, matching the other adapters. Images,
        # galleries and embeds are deliberately excluded from corpus text.
        paragraphs = [
            _html_text(e.get("content") or "")
            for e in (gc.get("content_elements") or [])
            if e.get("type") == "text"
        ]
        body = "\n".join(p for p in paragraphs if p)

        title = (gc.get("headlines") or {}).get("basic") or ""
        published = gc.get("first_publish_date") or gc.get("publish_date")
        modified = gc.get("last_updated_date")

        tax = gc.get("taxonomy") or {}
        primary = (tax.get("primary_section") or {}).get("name")
        categories = [
            s.get("name") for s in (tax.get("sections") or [])
            if isinstance(s, dict) and s.get("name")
        ]
        if primary and primary not in categories:
            categories.insert(0, primary)

        author = None
        by = (gc.get("credits") or {}).get("by") or []
        if by and isinstance(by[0], dict):
            author = by[0].get("name")

        if body or title:
            return record(
                url=url,
                outlet=outlet,
                title=title,
                body=body,
                published_date=published,
                category=primary,
                post_id=gc.get("_id") or _slug(url),
                categories=categories,
                modified_date=modified,
                author=author,
                language=gc.get("language"),
                tags=[t.get("text") for t in (tax.get("tags") or [])
                      if isinstance(t, dict) and t.get("text")],
            )

    # Fallback: JSON-LD plus the rendered article body.
    soup = BeautifulSoup(html, "lxml")
    news = None
    for obj in _iter_jsonld(soup):
        if obj.get("@type") == "NewsArticle":
            news = obj
            break
    if news is None:
        return None

    container = soup.find("div", class_="b-article-body")
    body = ""
    if container is not None:
        for junk in container.find_all(["script", "style", "noscript"]):
            junk.decompose()
        body = container.get_text("\n", strip=False)

    author = news.get("author")
    if isinstance(author, list) and author:
        a0 = author[0]
        author = a0.get("name") if isinstance(a0, dict) else str(a0)
    elif isinstance(author, dict):
        author = author.get("name")

    # Section is not in RFA's JSON-LD, so derive it from the URL path.
    m = re.search(r"/burmese/([a-z0-9_]+)/", url)
    section = m.group(1) if m else None

    return record(
        url=url,
        outlet=outlet,
        title=news.get("headline") or "",
        body=body,
        published_date=news.get("datePublished"),
        category=section,
        post_id=_slug(url),
        categories=[section] if section else [],
        modified_date=news.get("dateModified"),
        author=author,
        language=None,
        tags=[],
    )


def _slug(url: str) -> str:
    return url.rstrip("/").split("/")[-1].removesuffix(".html")
