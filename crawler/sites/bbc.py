"""
BBC Burmese -- https://www.bbc.com/burmese

Stack: Next.js (pages router). All article data lives in the __NEXT_DATA__ JSON
blob, which is far more reliable than scraping the rendered HTML.

robots.txt: /burmese/ is allowed (the 49 Disallow rules cover search and other
sections). No Crawl-delay is published, so we keep DVB's 2s out of courtesy.

Discovery is the hard part, and is why this adapter exists at all:

  - /burmese/sitemap.xml is STALE -- 100 URLs, all from September 2014.
  - The RSS feed is a rolling 20-item window: fine for updates, useless for
    backfill.
  - Topic listings hard-cap at 8 pages (~185 articles) and always show the most
    recent items, so busy topics cannot reach back to Jan 2025.

So the site's own navigation cannot cover the date range. We use the Wayback
CDX index purely as a URL *index* (~6,700 article URLs for Jan 2025 onward) and
then fetch every article live from bbc.com, so stored content is current.
Topic pages are walked afterwards to pick up anything too recent to be archived.

Parsing notes, all inverted from DVB:
  - No JSON-LD anywhere.
  - <h1> IS the real title here (on DVB it is sidebar chrome).
  - Dates are epoch milliseconds, not ISO strings.
"""

from __future__ import annotations

import json
import pathlib
import re
from datetime import datetime, timezone
from typing import Iterator

from ..core import Context, record

name = "bbc"
outlet = "bbc_burmese"
base_url = "https://www.bbc.com"
delay = 2.0          # no published Crawl-delay; matching DVB's out of courtesy

# NOTE: no from=/to= here. Those filter on CAPTURE date, not publish date, so
# they happily return a 2023 article that happened to be archived in 2025. We
# ask for every capture and filter client-side on the EARLIEST one instead.
CDX = (
    "http://web.archive.org/cdx/search/cdx"
    "?url=bbc.com/burmese/articles*&output=json&fl=original,timestamp"
    "&collapse=urlkey&limit=200000"
)
# BBC CPS ids are 'c' + 10-14 alphanumerics (e.g. c9vgyj5llx8jo). The CDX
# wildcard also matches URL fragments and asset paths -- roughly two thirds of
# its rows are junk like '000c' or '0049e072', all of which 404 on live bbc.com.
# Anchoring on the real id shape is what keeps the queue clean.
CPS_ID = r"c[a-z0-9]{10,14}"
ARTICLE_RE = re.compile(
    rf"^https?://(?:www\.)?bbc\.(?:com|co\.uk)/burmese/articles/({CPS_ID})$"
)
# Before July 2022 BBC Burmese used bare numeric story ids (/burmese/57296161)
# rather than /burmese/articles/<cps-id>. Those URLs are still live on bbc.com
# and are served by the SAME Next.js app, so parse() handles them unchanged --
# verified on ids from 2016, 2020, 2021 and 2022. Only this URL check had to
# learn about them. Anchoring at end-of-string excludes the '.amp' duplicates
# the CDX index is full of.
# Two legacy spellings exist: a bare id (/burmese/57296161) and a
# section-prefixed one (/burmese/burma-37579100, world-, media-, in-depth-,
# sport-). Both resolve on live bbc.com and both parse. The section prefix is
# matched generically rather than from a fixed list, since the list of BBC
# section slugs over 2016-2022 is not something we can enumerate reliably.
LEGACY_ID = r"(?:[a-z][a-z-]*-)?[0-9]{7,9}"
LEGACY_RE = re.compile(
    rf"^https?://(?:www\.)?bbc\.(?:com|co\.uk)/burmese/({LEGACY_ID})$"
)
# URL list enumerated from the Wayback CDX index (see .scratch/bbc_cdx/).
# Kept as a file because the pre-2022 ids are not densely packed -- guessing
# them returns 404 -- so the archive is the only way to learn which exist.
SEED_FILE = ".scratch/bbc_cdx/bbc_seed_urls.txt"
TOPIC_RE = re.compile(rf"/burmese/topics/({CPS_ID})")
ART_HREF_RE = re.compile(rf"/burmese/articles/({CPS_ID})")
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S
)

TOPIC_PAGE_CAP = 8   # measured: page 9 returns nothing


def _canonical(url: str) -> str | None:
    """Normalise to a canonical www.bbc.com article URL, or None if not one."""
    bare = url.split("?")[0].rstrip("/")
    m = ARTICLE_RE.match(bare)
    if m:
        return f"{base_url}/burmese/articles/{m.group(1)}"
    m = LEGACY_RE.match(bare)
    if m:
        return f"{base_url}/burmese/{m.group(1)}"
    return None


def _from_cdx(ctx: Context) -> Iterator[str]:
    if ctx.state.get("cdx_done"):
        ctx.log.info("bbc: CDX index already consumed")
        return

    ctx.log.info("bbc: querying Wayback CDX index (this takes ~30-60s)")

    status, body, err = ctx.fetch(CDX)
    if status != 200 or not body:
        ctx.log.error("bbc: CDX query failed (%s) -- falling back to topics only",
                      err or status)
        return

    try:
        rows = json.loads(body.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        ctx.log.error("bbc: CDX returned unparseable JSON")
        return

    # Earliest capture per article id.
    first: dict[str, str] = {}
    for row in rows[1:]:                    # row 0 is the header
        if len(row) < 2:
            continue
        canon = _canonical(row[0])
        if not canon:
            continue
        ts = row[1]
        if canon not in first or ts < first[canon]:
            first[canon] = ts

    # A page cannot be archived before it is published, so first_capture is
    # always >= published. Filtering on it therefore never drops an in-range
    # article; it only lets some older ones through, which the real date check
    # in core.run_fetch then skips.
    cutoff = ctx.start.replace("-", "")
    kept = 0
    for url, ts in sorted(first.items(), key=lambda kv: kv[1], reverse=True):
        if ts >= cutoff:
            kept += 1
            yield url

    ctx.log.info("bbc: CDX had %d article URLs, %d first-captured on/after %s",
                 len(first), kept, ctx.start)
    ctx.state.set("cdx_done", True)


def _topics(ctx: Context) -> list[str]:
    cached = ctx.state.get("topics")
    if cached:
        return cached
    status, body, _ = ctx.fetch(f"{base_url}/burmese")
    if status != 200 or not body:
        return []
    topics = sorted(set(TOPIC_RE.findall(body.decode("utf-8", "replace"))))
    ctx.state.set("topics", topics)
    ctx.log.info("bbc: %d topics discovered", len(topics))
    return topics


def _from_topics(ctx: Context) -> Iterator[str]:
    """Catch articles published too recently to be in the Wayback index."""
    cap = min(ctx.max_pages, TOPIC_PAGE_CAP) if ctx.max_pages else TOPIC_PAGE_CAP

    for topic in _topics(ctx):
        key = f"topic:{topic}"
        st = ctx.state.get(key, {"last_page": 0, "exhausted": False})
        if st["exhausted"]:
            continue

        for page in range(st["last_page"] + 1, cap + 1):
            status, body, _ = ctx.fetch(
                f"{base_url}/burmese/topics/{topic}?page={page}"
            )
            if status != 200 or not body:
                break

            ids = sorted(set(ART_HREF_RE.findall(body.decode("utf-8", "replace"))))
            if not ids:
                break

            for aid in ids:
                yield f"{base_url}/burmese/articles/{aid}"

            st["last_page"] = page
            ctx.state.set(key, st)
            ctx.log.info("bbc: topic %s p%d: %d articles", topic, page, len(ids))

        st["exhausted"] = True
        ctx.state.set(key, st)


def _from_seed(ctx: Context) -> Iterator[str]:
    """Yield URLs from a pre-enumerated list, if one has been prepared.

    The adapter's own CDX query only asks for /burmese/articles*, which cannot
    see the pre-2022 numeric ids. Enumerating those needs a full walk of the
    CDX index, which is slow and flaky enough to be worth doing once, offline,
    and checking in the result rather than repeating it on every run.
    """
    path = pathlib.Path(SEED_FILE)
    if not path.exists():
        return
    seen = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        canon = _canonical(line)
        if canon:
            seen += 1
            yield canon
    ctx.log.info("bbc: %d URLs from seed file %s", seen, SEED_FILE)


def discover(ctx: Context) -> Iterator[str]:
    yield from _from_seed(ctx)
    yield from _from_cdx(ctx)
    yield from _from_topics(ctx)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _fragments(node, acc: list[str]) -> list[str]:
    """Collect every text fragment under a content block, in document order."""
    if isinstance(node, dict):
        if node.get("type") == "fragment":
            text = node.get("model", {}).get("text")
            if isinstance(text, str):
                acc.append(text)
        for v in node.values():
            _fragments(v, acc)
    elif isinstance(node, list):
        for v in node:
            _fragments(v, acc)
    return acc


def parse(html: bytes, url: str) -> dict | None:
    m = NEXT_DATA_RE.search(html.decode("utf-8", "replace"))
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None

    page = data.get("props", {}).get("pageProps", {}).get("pageData")
    if not isinstance(page, dict):
        return None

    meta = page.get("metadata", {}) or {}
    blocks = page.get("content", {}).get("model", {}).get("blocks", []) or []
    if not blocks:
        return None

    # Body: paragraph blocks only, one per line. Images, promos, related links
    # and disclaimers are deliberately left out of the corpus text.
    paragraphs: list[str] = []
    for b in blocks:
        if b.get("type") == "text":
            frags = _fragments(b, [])
            if frags:
                paragraphs.append("".join(frags))
    body = "\n".join(paragraphs)

    # headline is often empty on this site; seoHeadline is the reliable one.
    heads = page.get("promo", {}).get("headlines", {}) or {}
    title = heads.get("headline") or heads.get("seoHeadline") or ""
    if not title:
        for b in blocks:
            if b.get("type") == "headline":
                title = "".join(_fragments(b, []))
                break

    # Epoch milliseconds -> ISO-8601 UTC, to match DVB's field format.
    published = None
    ts = meta.get("firstPublished")
    if isinstance(ts, (int, float)):
        published = datetime.fromtimestamp(ts / 1000, timezone.utc).isoformat()
    modified = None
    ts2 = meta.get("lastPublished")
    if isinstance(ts2, (int, float)):
        modified = datetime.fromtimestamp(ts2 / 1000, timezone.utc).isoformat()

    topics = [
        t.get("topicName") for t in (meta.get("topics") or [])
        if isinstance(t, dict) and t.get("topicName")
    ]

    byline = None
    for b in blocks:
        if b.get("type") == "byline":
            frags = _fragments(b, [])
            if frags:
                byline = frags[0]
            break

    aid = url.rstrip("/").split("/")[-1]

    return record(
        url=url,
        outlet=outlet,
        title=title,
        body=body,
        published_date=published,
        category=topics[0] if topics else None,
        post_id=aid,
        categories=topics,
        modified_date=modified,
        author=byline,
        language=meta.get("language"),
    )
