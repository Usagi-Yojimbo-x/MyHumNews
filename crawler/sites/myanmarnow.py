"""
Myanmar Now, Burmese edition -- https://myanmar-now.org/mm/

Stack: WordPress (Yoast), no CDN challenge. Plain python-requests works, so
this uses the default transport.

robots.txt is a Yoast block with `Disallow:` (empty -- allow everything) and no
Crawl-delay. We use 2s.

Discovery: the WordPress REST API is OPEN, which makes this the cleanest of the
five outlets.

    /mm/wp-json/wp/v2/posts?after=<iso>&before=<iso>&per_page=100&page=N

Native date filtering, native pagination, and an X-WP-Total header giving the
exact corpus size up front (2,455 for Jan 2025 onward). About 25 requests to
enumerate the whole range.

Crucially the API returns each post's canonical `link`, so unlike The Irrawaddy
there is no URL pattern to guess and no risk of a too-narrow regex silently
dropping most of the site. Whatever the API says the URL is, we use.

We deliberately do NOT take article bodies from the API even though
`content.rendered` carries them. Fetching each article's HTML keeps the raw
layer consistent with the other four outlets (gzipped HTML, re-parseable
offline) and keeps one parsing path. The API could supply bodies in ~25
requests instead of ~2,455 if a much faster re-run is ever wanted -- see the
README note.

Parsing matches The Irrawaddy closely: JSON-LD `Article` inside an `@graph`,
body in div.entry-content. As there, `inLanguage` claims en-US on Burmese
content and should not be trusted.
"""

from __future__ import annotations

import json
import re
from typing import Iterator

from bs4 import BeautifulSoup

from ..core import Context, record

name = "myanmarnow"
outlet = "myanmar_now_burmese"
base_url = "https://myanmar-now.org/mm"
delay = 2.0

API = f"{base_url}/wp-json/wp/v2/posts"
PER_PAGE = 100
LINK_RE = re.compile(
    r"https?://myanmar-now\.org/mm/([a-z][a-z-]*(?:/[a-z][a-z-]*)?)/(\d+)/?$"
)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def _api_url(ctx: Context, page: int) -> str:
    after = f"{ctx.start}T00:00:00"
    before = f"{ctx.end}T23:59:59" if ctx.end else None
    parts = [
        f"per_page={PER_PAGE}",
        f"page={page}",
        f"after={after}",
        "orderby=date",
        "order=desc",
        "_fields=id,date,link",
    ]
    if before:
        parts.append(f"before={before}")
    return API + "?" + "&".join(parts)


def discover(ctx: Context) -> Iterator[str]:
    page = ctx.state.get("api_page", 0) + 1
    seen_total = ctx.state.get("api_seen", 0)

    while True:
        if ctx.max_pages and page > ctx.max_pages:
            ctx.log.info("myanmarnow: hit --max-pages %d", ctx.max_pages)
            break

        status, body, err = ctx.fetch(_api_url(ctx, page))

        # The WP API returns 400 (rest_post_invalid_page_number) once you ask
        # for a page past the end -- that is the normal termination signal.
        if status == 400:
            ctx.log.info("myanmarnow: reached end of results at page %d", page)
            break
        if status != 200 or not body:
            ctx.log.warning("myanmarnow: page %d failed (%s), will retry next run",
                            page, err or status)
            break

        try:
            posts = json.loads(body.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            ctx.log.error("myanmarnow: page %d returned unparseable JSON", page)
            break

        if not isinstance(posts, list) or not posts:
            ctx.log.info("myanmarnow: page %d empty, done", page)
            break

        for p in posts:
            link = p.get("link")
            if link:
                # Trust the API's canonical link rather than reconstructing it.
                yield link

        seen_total += len(posts)
        ctx.state.set("api_page", page)
        ctx.state.set("api_seen", seen_total)
        ctx.log.info("myanmarnow: api page %d -> %d posts (%d total)",
                     page, len(posts), seen_total)

        if len(posts) < PER_PAGE:
            ctx.log.info("myanmarnow: short page, done")
            break
        page += 1


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _iter_jsonld(soup: BeautifulSoup):
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        items = data.get("@graph") if isinstance(data, dict) and "@graph" in data \
            else (data if isinstance(data, list) else [data])
        for obj in items:
            if isinstance(obj, dict):
                yield obj


def parse(html: bytes, url: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")

    art = None
    for obj in _iter_jsonld(soup):
        t = obj.get("@type")
        types = t if isinstance(t, list) else [t]
        if any("Article" in str(x) for x in types):
            art = obj
            break
    if art is None:
        return None

    body = ""
    container = soup.find("div", class_="entry-content")
    if container is not None:
        for junk in container.find_all(["script", "style", "noscript", "figure"]):
            junk.decompose()
        body = container.get_text("\n", strip=False)

    title = art.get("headline") or ""
    if not title:
        h1 = soup.find("h1", class_="entry-title")
        if h1:
            title = h1.get_text(strip=True)

    sec = art.get("articleSection")
    categories = [s for s in (sec if isinstance(sec, list) else [sec]) if s]

    author = art.get("author")
    if isinstance(author, list) and author:
        a0 = author[0]
        author = a0.get("name") if isinstance(a0, dict) else str(a0)
    elif isinstance(author, dict):
        author = author.get("name")

    m = LINK_RE.match(url.split("?")[0])
    section = m.group(1) if m else None
    post_id = m.group(2) if m else url.rstrip("/").rsplit("/", 1)[-1]

    # inLanguage reports en-US on Burmese content here, as on The Irrawaddy.
    return record(
        url=url,
        outlet=outlet,
        title=title,
        body=body,
        published_date=art.get("datePublished"),
        category=categories[0] if categories else None,
        post_id=post_id,
        categories=categories,
        modified_date=art.get("dateModified"),
        author=author,
        language=art.get("inLanguage"),
        section=section,
    )
