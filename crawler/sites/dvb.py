"""
DVB Burmese -- https://burmese.dvb.no

Stack: Next.js App Router + Strapi behind Cloudflare. Server-rendered, so no JS.

robots.txt: Allow /, Disallow /api/, Crawl-delay 2. The Strapi REST API under
/api/ is disallowed and is not used.

Discovery: there is no usable date index. The sitemap's <lastmod> is a bulk
migration timestamp, and post IDs are NOT chronological (941928 = Sept 2026,
999460 = June 2026). Category listings, however, ARE strictly date-descending,
so we walk those and stop each category once it pages past START_DATE.

Parsing: JSON-LD NewsArticle. Note <h1> is sidebar chrome ("About Us"), never
the headline, and articleSection carries only the PRIMARY category while
div.categories has the full list.
"""

from __future__ import annotations

import json
import re
from typing import Iterator

from bs4 import BeautifulSoup

from ..core import Context, record

name = "dvb"
outlet = "dvb_burmese"
base_url = "https://burmese.dvb.no"
delay = 2.0          # robots.txt Crawl-delay

MONTHS = {
    m: i + 1
    for i, m in enumerate(
        "January February March April May June July August September "
        "October November December".split()
    )
}
LISTING_DATE_RE = re.compile(
    r">(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{1,2}),\s*(\d{4})<"
)
POST_HREF_RE = re.compile(r'href="/post/(\d+)"')


def _categories(ctx: Context) -> list[str]:
    """Category slugs from sitemap shard 0. Percent-encoded Burmese slugs are
    listed there but 404 on the live route, so they are dropped."""
    cached = ctx.state.get("categories")
    if cached:
        return cached

    status, body, _ = ctx.fetch(f"{base_url}/sitemap/0.xml")
    if status != 200 or not body:
        ctx.log.error("dvb: could not load sitemap shard 0")
        return []

    text = body.decode("utf-8", "replace")
    slugs, seen = [], set()
    for loc in re.findall(r"<loc>([^<]+)</loc>", text):
        if "/categories/" not in loc:
            continue
        slug = loc.rsplit("/categories/", 1)[1].strip()
        if not slug or slug == "null" or "%" in slug or slug in seen:
            continue
        seen.add(slug)
        slugs.append(slug)

    ctx.state.set("categories", slugs)
    ctx.log.info("dvb: %d category slugs", len(slugs))
    return slugs


def _parse_listing(html: bytes) -> tuple[list[int], tuple[int, int, int] | None]:
    text = html.decode("utf-8", "replace")
    ids, seen = [], set()
    for pid in POST_HREF_RE.findall(text):
        n = int(pid)
        if n not in seen:
            seen.add(n)
            ids.append(n)
    dates = [
        (int(y), MONTHS[mo], int(d)) for mo, d, y in LISTING_DATE_RE.findall(text)
    ]
    return ids, (min(dates) if dates else None)


def discover(ctx: Context) -> Iterator[str]:
    start_t = ctx.start_tuple

    for slug in _categories(ctx):
        key = f"cat:{slug}"
        st = ctx.state.get(key, {"last_page": 0, "exhausted": False})
        if st["exhausted"]:
            ctx.log.info("dvb: category %s already exhausted", slug)
            continue

        page = st["last_page"] + 1
        while True:
            if ctx.max_pages and page > ctx.max_pages:
                ctx.log.info("dvb: %s hit --max-pages %d", slug, ctx.max_pages)
                break

            status, body, err = ctx.fetch(f"{base_url}/categories/{slug}?page={page}")
            if status != 200 or not body:
                st.update(exhausted=True)
                ctx.state.set(key, st)
                ctx.log.info("dvb: %s stopped at page %d (%s)", slug, page, err or status)
                break

            ids, oldest = _parse_listing(body)
            if not ids:
                st.update(exhausted=True)
                ctx.state.set(key, st)
                ctx.log.info("dvb: %s empty at page %d", slug, page)
                break

            for pid in ids:
                yield f"{base_url}/post/{pid}"

            st["last_page"] = page
            ctx.state.set(key, st)
            ctx.log.info("dvb: listing %s p%d: %d links, oldest=%s",
                         slug, page, len(ids), oldest)

            # Listings are strictly date-descending, so once this page's oldest
            # item predates the range, every later page does too.
            if oldest and oldest < start_t:
                st.update(exhausted=True)
                ctx.state.set(key, st)
                ctx.log.info("dvb: %s reached %s at page %d", slug, ctx.start, page)
                break

            page += 1


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
    soup = BeautifulSoup(html, "lxml")

    news = None
    for obj in _iter_jsonld(soup):
        if obj.get("@type") == "NewsArticle":
            news = obj
            break
    if news is None:
        return None

    body = ""
    container = soup.find("div", class_="full_content")
    if container is not None:
        for junk in container.find_all(["script", "style", "noscript"]):
            junk.decompose()
        body = container.get_text("\n", strip=False)

    primary = news.get("articleSection") or None
    categories: list[str] = []
    cat_div = soup.find("div", class_="categories")
    if cat_div is not None:
        for a in cat_div.find_all("a"):
            label = a.get_text(strip=True)
            if label and label not in categories:
                categories.append(label)
    if primary and primary not in categories:
        categories.insert(0, primary)

    author = news.get("author")
    if isinstance(author, dict):
        author = author.get("name")
    elif isinstance(author, list) and author:
        a0 = author[0]
        author = a0.get("name") if isinstance(a0, dict) else str(a0)

    m = re.search(r"/post/(\d+)", url)

    return record(
        url=url,
        outlet=outlet,
        title=news.get("headline") or "",
        body=body,
        published_date=news.get("datePublished"),
        category=primary,
        post_id=int(m.group(1)) if m else None,
        categories=categories,
        modified_date=news.get("dateModified"),
        author=author,
        language=news.get("inLanguage"),
    )
