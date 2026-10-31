#!/usr/bin/env python3
"""
DVB Burmese (https://burmese.dvb.no) news-corpus crawler.

Two phases, both resumable:

  discover  walk date-descending category listings, stop each category once it
            pages past START_DATE, and seed the SQLite work queue
  fetch     GET each pending article, gzip the raw HTML to disk, parse one
            JSONL record, mark done

Nothing is held in memory: raw HTML is gzipped straight to disk and each parsed
article is appended to the JSONL as soon as it is parsed.

Site notes (from recon, 2026-09-23):
  - Next.js + Strapi behind Cloudflare. Server-rendered, so no JS needed.
  - robots.txt: Allow /, Disallow /api/, Crawl-delay 2.
  - No RSS/Atom. sitemap.xml exists but its <lastmod> is a bulk-migration
    timestamp, and post IDs are NOT chronological -- neither can date-filter.
  - Category listings ARE strictly date-descending, so they are the only cheap
    way to reach "Jan 2025 onward" without fetching all 138,950 articles.
  - Article title lives in JSON-LD, NOT in <h1> (<h1> is sidebar chrome).
  - JSON-LD articleSection holds only the PRIMARY category; div.categories has
    the full list. Both are stored.

Burmese text is stored exactly as the site serves it. No Zawgyi/Unicode
conversion, no whitespace collapsing, no NFC normalization.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_URL = "https://burmese.dvb.no"
OUTLET = "dvb_burmese"

# Inclusive date range for the corpus (UTC, YYYY-MM-DD). END_DATE None = today.
START_DATE = "2025-01-01"
END_DATE = None

# Seconds between requests. robots.txt asks for 2.
REQUEST_DELAY = 2.0

OUTPUT_DIR = "data/dvb"

# Contact email goes in the User-Agent so the site can reach you.
CONTACT_EMAIL = "zwehtetaung.doh@gmail.com"
USER_AGENT = (
    "Mozilla/5.0 (compatible; MyHumNewsBot/0.1; "
    "university news-corpus research; +mailto:{email})"
)

MAX_ATTEMPTS = 3            # total tries per URL before marking failed
BACKOFF_BASE = 2.0          # sleep = BACKOFF_BASE ** attempt, on 429/5xx
REQUEST_TIMEOUT = 30
MAX_RETRY_AFTER = 120       # cap on an honoured Retry-After header

# Categories to walk. Empty list = auto-discover from sitemap shard 0.
# Dead/legacy slugs are skipped automatically when they 404.
CATEGORIES: list[str] = []

# ---------------------------------------------------------------------------

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

LOG = logging.getLogger("dvb")

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


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS urls (
    url            TEXT PRIMARY KEY,
    post_id        INTEGER,
    status         TEXT NOT NULL DEFAULT 'pending',   -- pending|done|failed|skipped
    attempts       INTEGER NOT NULL DEFAULT 0,
    http_status    INTEGER,
    published_date TEXT,
    error          TEXT,
    updated_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_urls_status ON urls(status);

CREATE TABLE IF NOT EXISTS listing_pages (
    category  TEXT NOT NULL,
    page      INTEGER NOT NULL,
    oldest    TEXT,
    n_links   INTEGER,
    fetched_at TEXT,
    PRIMARY KEY (category, page)
);

CREATE TABLE IF NOT EXISTS categories (
    slug       TEXT PRIMARY KEY,
    exhausted  INTEGER NOT NULL DEFAULT 0,  -- 1 = paged past START_DATE or dead
    last_page  INTEGER NOT NULL DEFAULT 0,
    note       TEXT
);

CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# polite HTTP
# ---------------------------------------------------------------------------

class Fetcher:
    """Rate-limited GET with exponential backoff on 429/5xx."""

    def __init__(self, delay: float, max_attempts: int):
        self.delay = delay
        self.max_attempts = max_attempts
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT.format(email=CONTACT_EMAIL),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "my,en;q=0.8",
            }
        )

    def _wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.delay:
            time.sleep(self.delay - gap)
        self._last = time.monotonic()

    def get(self, url: str) -> tuple[int | None, bytes | None, str | None]:
        """Return (http_status, body_bytes, error). Retries 429/5xx and
        connection errors; 4xx other than 429 fail immediately."""
        last_err = None
        last_status = None

        for attempt in range(1, self.max_attempts + 1):
            self._wait()
            t0 = time.monotonic()
            try:
                r = self.session.get(url, timeout=REQUEST_TIMEOUT)
                elapsed = time.monotonic() - t0
                last_status = r.status_code

                if r.status_code == 200:
                    LOG.info(
                        "GET %s -> 200 %d bytes in %.2fs (attempt %d)",
                        url, len(r.content), elapsed, attempt,
                    )
                    return 200, r.content, None

                retryable = r.status_code == 429 or 500 <= r.status_code < 600
                LOG.warning(
                    "GET %s -> %d in %.2fs (attempt %d/%d)%s",
                    url, r.status_code, elapsed, attempt, self.max_attempts,
                    "" if retryable else " [not retryable]",
                )
                last_err = f"HTTP {r.status_code}"
                if not retryable:
                    return r.status_code, None, last_err

                if attempt < self.max_attempts:
                    sleep_s = BACKOFF_BASE ** attempt
                    ra = r.headers.get("Retry-After")
                    if ra:
                        try:
                            sleep_s = max(sleep_s, min(float(ra), MAX_RETRY_AFTER))
                        except ValueError:
                            pass
                    LOG.info("backing off %.1fs before retry", sleep_s)
                    time.sleep(sleep_s)

            except requests.RequestException as e:
                elapsed = time.monotonic() - t0
                last_err = f"{type(e).__name__}: {e}"[:300]
                LOG.warning(
                    "GET %s -> ERROR in %.2fs (attempt %d/%d): %s",
                    url, elapsed, attempt, self.max_attempts, last_err,
                )
                if attempt < self.max_attempts:
                    time.sleep(BACKOFF_BASE ** attempt)

        LOG.error("GET %s giving up after %d attempts: %s",
                  url, self.max_attempts, last_err)
        return last_status, None, last_err


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
        for obj in data if isinstance(data, list) else [data]:
            if isinstance(obj, dict):
                yield obj


def parse_article(html: bytes, url: str) -> dict | None:
    """Extract one article record. Returns None if it isn't a NewsArticle."""
    soup = BeautifulSoup(html, "lxml")

    news = None
    for obj in _iter_jsonld(soup):
        if obj.get("@type") == "NewsArticle":
            news = obj
            break
    if news is None:
        return None

    # Title: JSON-LD only. <h1> on this site is sidebar chrome ("About Us").
    title = news.get("headline") or ""

    # Body: div.full_content. Inner nesting differs between site eras, so take
    # the container and drop non-text children rather than pinning a chain.
    body = ""
    container = soup.find("div", class_="full_content")
    if container is not None:
        for junk in container.find_all(["script", "style", "noscript"]):
            junk.decompose()
        body = container.get_text("\n", strip=False)

    # Categories: JSON-LD gives only the primary one; div.categories has all.
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
    post_id = int(m.group(1)) if m else None

    return {
        # the schema you asked for, in order
        "url": url,
        "outlet": OUTLET,
        "title": title,
        "body": body,
        "published_date": news.get("datePublished"),
        "category": primary,
        "scraped_at": now_iso(),
        "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        # extras
        "post_id": post_id,
        "categories": categories,
        "modified_date": news.get("dateModified"),
        "author": author,
        "language": news.get("inLanguage"),
    }


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def parse_listing(html: bytes) -> tuple[list[int], tuple[int, int, int] | None]:
    """Return (post ids on page, oldest date tuple seen)."""
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


def sitemap_categories(fetcher: Fetcher) -> list[str]:
    status, body, _ = fetcher.get(f"{BASE_URL}/sitemap/0.xml")
    if status != 200 or not body:
        LOG.error("could not load sitemap shard 0 for category discovery")
        return []
    text = body.decode("utf-8", "replace")
    slugs, seen = [], set()
    for loc in re.findall(r"<loc>([^<]+)</loc>", text):
        if "/categories/" not in loc:
            continue
        slug = loc.rsplit("/categories/", 1)[1].strip()
        # Percent-encoded Burmese slugs in the sitemap 404 on the live route.
        if not slug or slug == "null" or "%" in slug or slug in seen:
            continue
        seen.add(slug)
        slugs.append(slug)
    return slugs


def discover(conn, fetcher, start, end, max_pages, only_cats, url_limit):
    cats = only_cats or CATEGORIES or sitemap_categories(fetcher)
    if not cats:
        LOG.error("no categories to walk")
        return 0

    for slug in cats:
        conn.execute(
            "INSERT OR IGNORE INTO categories(slug) VALUES (?)", (slug,)
        )
    conn.commit()

    start_t = tuple(int(x) for x in start.split("-"))
    new_total = 0

    for slug in cats:
        row = conn.execute(
            "SELECT exhausted, last_page FROM categories WHERE slug=?", (slug,)
        ).fetchone()
        exhausted, last_page = (row if row else (0, 0))
        if exhausted:
            LOG.info("category %s already exhausted, skipping", slug)
            continue

        page = last_page + 1
        while True:
            if max_pages and page > max_pages:
                LOG.info("category %s hit --max-pages %d", slug, max_pages)
                break

            status, body, err = fetcher.get(
                f"{BASE_URL}/categories/{slug}?page={page}"
            )
            if status != 200 or not body:
                conn.execute(
                    "UPDATE categories SET exhausted=1, note=? WHERE slug=?",
                    (f"stopped at page {page}: {err or status}", slug),
                )
                conn.commit()
                LOG.info("category %s stopped at page %d (%s)", slug, page, err or status)
                break

            ids, oldest = parse_listing(body)
            if not ids:
                conn.execute(
                    "UPDATE categories SET exhausted=1, note=? WHERE slug=?",
                    (f"empty at page {page}", slug),
                )
                conn.commit()
                LOG.info("category %s empty at page %d", slug, page)
                break

            new_here = 0
            for pid in ids:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO urls(url, post_id, status, updated_at) "
                    "VALUES (?,?,'pending',?)",
                    (f"{BASE_URL}/post/{pid}", pid, now_iso()),
                )
                new_here += cur.rowcount
            new_total += new_here

            conn.execute(
                "INSERT OR REPLACE INTO listing_pages"
                "(category,page,oldest,n_links,fetched_at) VALUES (?,?,?,?,?)",
                (slug, page,
                 "-".join(f"{p:02d}" for p in oldest) if oldest else None,
                 len(ids), now_iso()),
            )
            conn.execute(
                "UPDATE categories SET last_page=? WHERE slug=?", (page, slug)
            )
            conn.commit()

            LOG.info(
                "listing %s p%d: %d links (%d new), oldest=%s",
                slug, page, len(ids), new_here, oldest,
            )

            if url_limit and new_total >= url_limit:
                LOG.info("reached --limit %d discovered URLs", url_limit)
                return new_total

            # Listings are strictly date-descending: once this page's oldest
            # item predates the range, every later page does too.
            if oldest and oldest < start_t:
                conn.execute(
                    "UPDATE categories SET exhausted=1, note=? WHERE slug=?",
                    (f"paged past {start} at page {page}", slug),
                )
                conn.commit()
                LOG.info("category %s reached %s at page %d", slug, start, page)
                break

            page += 1

    return new_total


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def in_range(published: str | None, start: str, end: str | None) -> bool:
    if not published:
        return False
    day = published[:10]
    if day < start:
        return False
    return not (end and day > end)


def fetch_all(conn, fetcher, paths, start, end, limit):
    raw_dir = paths["raw"]
    rows = conn.execute(
        "SELECT url, post_id FROM urls WHERE status='pending' ORDER BY post_id DESC"
        + (" LIMIT ?" if limit else ""),
        (limit,) if limit else (),
    ).fetchall()

    if not rows:
        LOG.info("nothing pending")
        return 0, 0, 0

    LOG.info("fetching %d pending article(s)", len(rows))
    done = failed = skipped = 0

    # Append mode: the JSONL is the durable output, the DB is the bookkeeping.
    with open(paths["jsonl"], "a", encoding="utf-8") as out:
        for url, post_id in rows:
            conn.execute(
                "UPDATE urls SET attempts=attempts+1, updated_at=? WHERE url=?",
                (now_iso(), url),
            )
            conn.commit()

            status, body, err = fetcher.get(url)

            if status != 200 or not body:
                attempts = conn.execute(
                    "SELECT attempts FROM urls WHERE url=?", (url,)
                ).fetchone()[0]
                # Fetcher already burned MAX_ATTEMPTS internally on retryable
                # errors; a non-retryable status is final straight away.
                final = True
                conn.execute(
                    "UPDATE urls SET status=?, http_status=?, error=?, updated_at=? "
                    "WHERE url=?",
                    ("failed" if final else "pending", status, err, now_iso(), url),
                )
                conn.commit()
                failed += 1
                continue

            # Raw layer first, so a parser change never needs a re-crawl.
            raw_path = os.path.join(raw_dir, f"{post_id}.html.gz")
            with gzip.open(raw_path, "wb", compresslevel=6) as gz:
                gz.write(body)

            try:
                rec = parse_article(body, url)
            except Exception as e:  # noqa: BLE001 - never lose the queue to one bad page
                LOG.exception("parse failed for %s", url)
                conn.execute(
                    "UPDATE urls SET status='failed', http_status=?, error=?, "
                    "updated_at=? WHERE url=?",
                    (status, f"parse: {type(e).__name__}: {e}"[:300], now_iso(), url),
                )
                conn.commit()
                failed += 1
                continue

            if rec is None:
                conn.execute(
                    "UPDATE urls SET status='failed', http_status=?, error=?, "
                    "updated_at=? WHERE url=?",
                    (status, "no NewsArticle JSON-LD", now_iso(), url),
                )
                conn.commit()
                failed += 1
                continue

            pub = rec["published_date"]
            if not in_range(pub, start, end):
                conn.execute(
                    "UPDATE urls SET status='skipped', http_status=?, "
                    "published_date=?, error=?, updated_at=? WHERE url=?",
                    (status, pub, "out of date range", now_iso(), url),
                )
                conn.commit()
                skipped += 1
                LOG.info("skip %s (published %s, outside range)", url, pub)
                continue

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())

            conn.execute(
                "UPDATE urls SET status='done', http_status=?, published_date=?, "
                "error=NULL, updated_at=? WHERE url=?",
                (status, pub, now_iso(), url),
            )
            conn.commit()
            done += 1

    return done, failed, skipped


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def setup_paths(outdir: str) -> dict:
    raw = os.path.join(outdir, "raw")
    parsed = os.path.join(outdir, "parsed")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(parsed, exist_ok=True)
    return {
        "root": outdir,
        "raw": raw,
        "jsonl": os.path.join(parsed, "articles.jsonl"),
        "db": os.path.join(outdir, "crawl.db"),
        "log": os.path.join(outdir, "crawl.log"),
    }


def setup_logging(path: str, quiet: bool) -> None:
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%dT%H:%M:%S%z"
    )
    LOG.setLevel(logging.INFO)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    LOG.addHandler(fh)
    if not quiet:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        LOG.addHandler(sh)


def cmd_status(conn) -> None:
    print("status counts:")
    for st, n in conn.execute(
        "SELECT status, COUNT(*) FROM urls GROUP BY status ORDER BY 2 DESC"
    ):
        print(f"  {st:8} {n}")
    row = conn.execute(
        "SELECT MIN(published_date), MAX(published_date) FROM urls "
        "WHERE status='done'"
    ).fetchone()
    if row and row[0]:
        print(f"  date range: {row[0][:10]} .. {row[1][:10]}")
    ex = conn.execute(
        "SELECT COUNT(*), SUM(exhausted) FROM categories"
    ).fetchone()
    if ex and ex[0]:
        print(f"  categories: {ex[1] or 0}/{ex[0]} exhausted")


def main() -> int:
    p = argparse.ArgumentParser(description="DVB Burmese corpus crawler")
    p.add_argument("phase", choices=["discover", "fetch", "run", "status"],
                   help="run = discover then fetch")
    p.add_argument("--out", default=OUTPUT_DIR)
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=END_DATE)
    p.add_argument("--delay", type=float, default=REQUEST_DELAY)
    p.add_argument("--limit", type=int, default=0,
                   help="cap URLs discovered / articles fetched (testing)")
    p.add_argument("--max-pages", type=int, default=0,
                   help="cap listing pages per category (testing)")
    p.add_argument("--categories", default="",
                   help="comma-separated slugs; default = all from sitemap")
    p.add_argument("--retry-failed", action="store_true",
                   help="reset failed URLs to pending before fetching")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    paths = setup_paths(args.out)
    setup_logging(paths["log"], args.quiet)
    conn = open_db(paths["db"])

    LOG.info("=== dvb_crawler %s | out=%s delay=%.1fs range=%s..%s ===",
             args.phase, args.out, args.delay, args.start, args.end or "today")

    if args.phase == "status":
        cmd_status(conn)
        return 0

    fetcher = Fetcher(args.delay, MAX_ATTEMPTS)
    only = [c.strip() for c in args.categories.split(",") if c.strip()]

    if args.retry_failed:
        n = conn.execute(
            "UPDATE urls SET status='pending', attempts=0 WHERE status='failed'"
        ).rowcount
        conn.commit()
        LOG.info("reset %d failed URL(s) to pending", n)

    if args.phase in ("discover", "run"):
        n = discover(conn, fetcher, args.start, args.end,
                     args.max_pages, only, args.limit)
        LOG.info("discovery added %d new URL(s)", n)

    if args.phase in ("fetch", "run"):
        done, failed, skipped = fetch_all(
            conn, fetcher, paths, args.start, args.end, args.limit
        )
        LOG.info("fetch complete: %d done, %d failed, %d skipped",
                 done, failed, skipped)

    cmd_status(conn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
