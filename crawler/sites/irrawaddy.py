"""
The Irrawaddy, Burmese edition -- https://burma.irrawaddy.com

Stack: WordPress (JNews theme) with Yoast SEO sitemaps, behind Cloudflare.

robots.txt is `User-agent: *` with NO Disallow lines at all -- crawling is
explicitly permitted, and there is no Crawl-delay. We use 2s regardless.

CLOUDFLARE: this is the only outlet that python-requests cannot fetch. The CDN
scores clients on their TLS/HTTP2 handshake as well as their headers, and
python-requests is challenged on every path while curl with a full browser
header set succeeds (measured 10/10 on articles, 12/12 across sitemaps,
category pages and articles). Identical requests seconds apart returned 200
from curl and 403 from requests, which is what identified the fingerprint as
the discriminator rather than the headers.

This adapter therefore declares `transport = "curl"` and core.CurlFetcher
handles it. See that class's docstring for the reasoning; in short, we are not
spoofing a browser fingerprint, we are using the HTTP client the CDN accepts,
on a site whose robots.txt invites crawling.

Note the interstitial is a 403 carrying an HTML challenge page rather than a
network error, so CurlFetcher treats it as retryable -- it frequently clears on
a later attempt, and appears to be an IP warm-up behaviour.

Discovery: sitemap_index.xml lists 226 sub-sitemaps, of which ~68 are
`post-sitemap*.xml` (1000 URLs each, ~68k articles). They are NOT ordered by
publish date -- post-sitemap68 spans 2014 to 2026 -- so we cannot walk
backwards and stop early. Instead we fetch all of them, which is cheap, and
filter on the date embedded in each article URL:

    /<section>/YYYY/MM/DD/<id>.html

where <section> is one or two path segments (/news/, /video/, /article/,
/opinion/viewpoint/, /news/short-news/, ...). Matching only /article/ captures
barely 12% of the site, so the pattern must stay general. That embedded date means we know an article's publish date BEFORE fetching it,
so nothing out of range is ever downloaded. Only DVB lacks this luxury.
"""

from __future__ import annotations

import json
import re
from typing import Iterator

from bs4 import BeautifulSoup

from ..core import Context, record

name = "irrawaddy"
outlet = "irrawaddy_burmese"
base_url = "https://burma.irrawaddy.com"
delay = 2.0
transport = "curl"          # see module docstring / core.CurlFetcher

SITEMAP_INDEX = f"{base_url}/sitemap_index.xml"
LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")
POST_SITEMAP_RE = re.compile(r"/post-sitemap\d*\.xml$")
# Articles live under MANY section prefixes, not just /article/ -- measured on
# one sitemap: 196 /video/, 125 /news/, 87 /article/, plus /opinion/viewpoint/,
# /opinion/interview/, /opinion/editorial/, /news/short-news/, /cartoon/,
# /lifestyle/, /business/. Sections can be one OR two path segments, and an
# /article/-only pattern captures barely 12% of the site. The section is
# captured so it can be recorded and filtered on later.
ARTICLE_RE = re.compile(
    r"https?://burma\.irrawaddy\.com/"
    r"([a-z][a-z-]*(?:/[a-z][a-z-]*)?)/"
    r"(\d{4})/(\d{2})/(\d{2})/(\d+)\.html"
)

# Sections that are not primarily prose. Kept in the corpus by default (they
# are published content) but recorded so they can be excluded from text work.
NON_PROSE_SECTIONS = {"video", "cartoon", "photo", "gallery"}


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def _post_sitemaps(ctx: Context) -> list[str]:
    cached = ctx.state.get("post_sitemaps")
    if cached:
        return cached

    status, body, err = ctx.fetch(SITEMAP_INDEX)
    if status != 200 or not body:
        ctx.log.error("irrawaddy: sitemap index failed (%s)", err or status)
        return []

    locs = LOC_RE.findall(body.decode("utf-8", "replace"))
    maps = [u for u in locs if POST_SITEMAP_RE.search(u)]
    ctx.state.set("post_sitemaps", maps)
    ctx.log.info("irrawaddy: %d post-sitemaps (of %d total)", len(maps), len(locs))
    return maps


def discover(ctx: Context) -> Iterator[str]:
    maps = _post_sitemaps(ctx)
    if not maps:
        return

    done = set(ctx.state.get("sitemaps_done", []))
    in_range = ctx.state.get("in_range_count", 0)

    for i, sm in enumerate(maps, 1):
        if sm in done:
            continue
        if ctx.max_pages and i > ctx.max_pages:
            ctx.log.info("irrawaddy: hit --max-pages %d sitemaps", ctx.max_pages)
            break

        status, body, err = ctx.fetch(sm)
        if status != 200 or not body:
            # Leave it out of `done` so the next run retries it.
            ctx.log.warning("irrawaddy: %s failed (%s), will retry next run",
                            sm.rsplit("/", 1)[-1], err or status)
            continue

        found = 0
        for sec, y, mo, d, pid in ARTICLE_RE.findall(
                body.decode("utf-8", "replace")):
            day = f"{y}-{mo}-{d}"
            # The publish date is in the URL, so out-of-range articles are
            # filtered here and never fetched at all.
            if day < ctx.start:
                continue
            if ctx.end and day > ctx.end:
                continue
            found += 1
            yield f"{base_url}/{sec}/{y}/{mo}/{d}/{pid}.html"

        in_range += found
        done.add(sm)
        ctx.state.set("sitemaps_done", sorted(done))
        ctx.state.set("in_range_count", in_range)
        ctx.log.info("irrawaddy: %s -> %d in-range (%d/%d sitemaps, %d total)",
                     sm.rsplit("/", 1)[-1], found, len(done), len(maps), in_range)


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

    # Body: JNews renders the article into div.entry-content.
    body = ""
    container = soup.find("div", class_="entry-content")
    if container is not None:
        for junk in container.find_all(["script", "style", "noscript", "figure"]):
            junk.decompose()
        body = container.get_text("\n", strip=False)

    title = art.get("headline") or ""
    if not title:
        h1 = soup.find("h1", class_="jeg_post_title")
        if h1:
            title = h1.get_text(strip=True)

    # articleSection is already a list here (unlike DVB, where it is a single
    # string), and mixes English and Burmese labels.
    sec = art.get("articleSection")
    categories = [s for s in (sec if isinstance(sec, list) else [sec]) if s]

    author = art.get("author")
    if isinstance(author, list) and author:
        a0 = author[0]
        author = a0.get("name") if isinstance(a0, dict) else str(a0)
    elif isinstance(author, dict):
        author = author.get("name")

    m = ARTICLE_RE.search(url)
    post_id = m.group(5) if m else url.rsplit("/", 1)[-1].removesuffix(".html")
    section = m.group(1) if m else None

    # WordPress reports inLanguage as en-US on this site even though the
    # content is Burmese, so it is recorded but should not be trusted.
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
        is_prose=(section or "").split("/")[0] not in NON_PROSE_SECTIONS,
    )
