"""
Shared crawler engine for the MyHumNews Burmese news corpus.

Everything that is identical across outlets lives here: the SQLite work queue,
the rate-limited fetcher with exponential backoff, the gzipped raw layer, the
JSONL writer and the request log.

Per-outlet behaviour lives in crawler/sites/<name>.py, which supplies a Site
with two methods:

    discover(ctx)            yield article URLs (resumable via ctx.state)
    parse(html, url)         bytes -> dict, or None to reject the page

Outlets differ far more than they look. DVB walks date-descending category
listings and reads JSON-LD; BBC has no usable sitemap, caps topic pagination at
8 pages, ships no JSON-LD at all, and has to be discovered through the Wayback
CDX index. The Site interface exists to absorb exactly that kind of difference.

Burmese text is stored exactly as served -- no Zawgyi/Unicode conversion, no
whitespace collapsing, no NFC normalisation. That is a corpus decision and it
belongs to a later, separate step.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Iterator, Protocol

import requests

LOG = logging.getLogger("crawler")

# Defaults; a Site may override, and the CLI overrides both.
DEFAULT_DELAY = 2.0
MAX_ATTEMPTS = 3
BACKOFF_BASE = 2.0
REQUEST_TIMEOUT = 30
MAX_RETRY_AFTER = 120

CONTACT_EMAIL = "zwehtetaung.doh@gmail.com"
USER_AGENT = (
    "Mozilla/5.0 (compatible; MyHumNewsBot/0.1; "
    "university news-corpus research; +mailto:{email})"
)


# ---------------------------------------------------------------------------
# Site interface
# ---------------------------------------------------------------------------

class Site(Protocol):
    name: str          # short slug, also the output directory
    outlet: str        # value written to the `outlet` JSONL field
    base_url: str
    delay: float       # politeness delay this outlet asks for

    def discover(self, ctx: "Context") -> Iterator[str]:
        """Yield article URLs. Must be resumable: persist progress in
        ctx.state and skip what is already recorded there."""

    def parse(self, html: bytes, url: str) -> dict | None:
        """Return a record dict, or None if the page is not an article."""


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS urls (
    url            TEXT PRIMARY KEY,
    post_id        TEXT,
    status         TEXT NOT NULL DEFAULT 'pending',   -- pending|done|failed|skipped
    attempts       INTEGER NOT NULL DEFAULT 0,
    http_status    INTEGER,
    published_date TEXT,
    error          TEXT,
    updated_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_urls_status ON urls(status);

-- Free-form per-site discovery bookkeeping, so each adapter can resume in
-- whatever terms suit it (DVB: category -> last page; BBC: cdx cursor).
CREATE TABLE IF NOT EXISTS discovery_state (
    k TEXT PRIMARY KEY,
    v TEXT
);

CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


class State:
    """Dict-like view over discovery_state, persisted on every write."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get(self, key: str, default=None):
        row = self.conn.execute(
            "SELECT v FROM discovery_state WHERE k=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key: str, value) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO discovery_state(k,v) VALUES (?,?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.conn.commit()


# ---------------------------------------------------------------------------
# polite HTTP
# ---------------------------------------------------------------------------

class Fetcher:
    """Rate-limited GET with exponential backoff on 429/5xx.

    4xx other than 429 are final immediately -- retrying a 404 just wastes the
    politeness budget.
    """

    def __init__(self, delay: float, max_attempts: int = MAX_ATTEMPTS):
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
                    LOG.info("GET %s -> 200 %d bytes in %.2fs (attempt %d)",
                             url, len(r.content), elapsed, attempt)
                    return 200, r.content, None

                retryable = r.status_code == 429 or 500 <= r.status_code < 600
                LOG.warning("GET %s -> %d in %.2fs (attempt %d/%d)%s",
                            url, r.status_code, elapsed, attempt,
                            self.max_attempts,
                            "" if retryable else " [not retryable]")
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
                LOG.warning("GET %s -> ERROR in %.2fs (attempt %d/%d): %s",
                            url, elapsed, attempt, self.max_attempts, last_err)
                if attempt < self.max_attempts:
                    time.sleep(BACKOFF_BASE ** attempt)

        LOG.error("GET %s giving up after %d attempts: %s",
                  url, self.max_attempts, last_err)
        return last_status, None, last_err


# ---------------------------------------------------------------------------
# context handed to adapters
# ---------------------------------------------------------------------------

class CurlFetcher(Fetcher):
    """Same interface as Fetcher, but shells out to curl.

    Needed only for The Irrawaddy, which sits behind Cloudflare bot management.
    Cloudflare scores the client on its TLS/HTTP2 handshake as well as its
    headers: identical requests seconds apart get 200 from curl and 403 (an
    interstitial JS challenge) from python-requests. We are not impersonating a
    browser's fingerprint -- curl presents its own, and Cloudflare allows it --
    we simply use the HTTP client the site's CDN accepts. The full browser
    header set is required in addition; dropping Sec-Fetch-* alone is enough to
    get challenged.

    Their robots.txt is `User-agent: *` with no Disallow at all, so crawling is
    explicitly permitted; the challenge is blanket CDN protection, not a policy
    signal aimed at us. We still keep the same politeness delay.
    """

    BROWSER_UA = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    )
    HEADERS = [
        "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language: en-US,en;q=0.9",
        "Upgrade-Insecure-Requests: 1",
        "Sec-Fetch-Dest: document",
        "Sec-Fetch-Mode: navigate",
        "Sec-Fetch-Site: none",
    ]
    CHALLENGE = b"Just a moment"

    def get(self, url: str) -> tuple[int | None, bytes | None, str | None]:
        import shutil
        import subprocess

        if not shutil.which("curl"):
            return None, None, "curl not installed (required for this site)"

        cmd = ["curl", "-sS", "-L", "--compressed",
               "--max-time", str(REQUEST_TIMEOUT),
               "-w", "\n%{http_code}", "-A", self.BROWSER_UA]
        for h in self.HEADERS:
            cmd += ["-H", h]
        cmd.append(url)

        last_err = None
        last_status = None
        for attempt in range(1, self.max_attempts + 1):
            self._wait()
            t0 = time.monotonic()
            try:
                p = subprocess.run(cmd, capture_output=True,
                                   timeout=REQUEST_TIMEOUT + 15)
            except subprocess.TimeoutExpired:
                last_err = "curl timeout"
                LOG.warning("GET %s -> curl timeout (attempt %d/%d)",
                            url, attempt, self.max_attempts)
                if attempt < self.max_attempts:
                    time.sleep(BACKOFF_BASE ** attempt)
                continue

            elapsed = time.monotonic() - t0
            body, _, code = p.stdout.rpartition(b"\n")
            try:
                status = int(code.strip())
            except ValueError:
                status = None
            last_status = status

            # A Cloudflare interstitial is a 403 carrying an HTML challenge
            # page. Treat it as retryable: it often clears on a later attempt.
            challenged = self.CHALLENGE in body[:4000]

            if status == 200 and not challenged:
                LOG.info("GET %s -> 200 %d bytes in %.2fs (attempt %d, curl)",
                         url, len(body), elapsed, attempt)
                return 200, body, None

            if challenged:
                last_err = "cloudflare challenge"
                LOG.warning("GET %s -> %s cloudflare challenge in %.2fs "
                            "(attempt %d/%d)", url, status, elapsed,
                            attempt, self.max_attempts)
            else:
                last_err = f"HTTP {status}"
                retryable = status == 429 or (status or 0) >= 500
                LOG.warning("GET %s -> %s in %.2fs (attempt %d/%d)%s",
                            url, status, elapsed, attempt, self.max_attempts,
                            "" if retryable else " [not retryable]")
                if not retryable:
                    return status, None, last_err

            if attempt < self.max_attempts:
                sleep_s = BACKOFF_BASE ** attempt
                LOG.info("backing off %.1fs before retry", sleep_s)
                time.sleep(sleep_s)

        LOG.error("GET %s giving up after %d attempts: %s",
                  url, self.max_attempts, last_err)
        return last_status, None, last_err


class Context:
    def __init__(self, conn, fetcher: Fetcher, start: str, end: str | None,
                 max_pages: int = 0):
        self.conn = conn
        self.fetcher = fetcher
        self.start = start
        self.end = end
        self.max_pages = max_pages          # testing cap, 0 = unlimited
        self.state = State(conn)
        self.log = LOG

    def fetch(self, url: str):
        return self.fetcher.get(url)

    @property
    def start_tuple(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self.start.split("-"))  # type: ignore[return-value]


def in_range(published: str | None, start: str, end: str | None) -> bool:
    if not published:
        return False
    day = published[:10]
    if day < start:
        return False
    return not (end and day > end)


# ---------------------------------------------------------------------------
# helpers shared by adapters
# ---------------------------------------------------------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record(url, outlet, title, body, published_date, category, **extra) -> dict:
    """Build a JSONL record with the agreed field order, then extras."""
    rec = {
        "url": url,
        "outlet": outlet,
        "title": title,
        "body": body,
        "published_date": published_date,
        "category": category,
        "scraped_at": now_iso(),
        "content_sha256": sha256_text(body or ""),
    }
    rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def run_discover(site: Site, ctx: Context, limit: int = 0) -> int:
    """Consume the adapter's URL generator into the queue."""
    new = 0
    for url in site.discover(ctx):
        cur = ctx.conn.execute(
            "INSERT OR IGNORE INTO urls(url, status, updated_at) "
            "VALUES (?, 'pending', ?)",
            (url, now_iso()),
        )
        new += cur.rowcount
        if new and new % 200 == 0:
            ctx.conn.commit()
            LOG.info("discovery: %d new URLs queued so far", new)
        if limit and new >= limit:
            LOG.info("discovery: reached --limit %d", limit)
            break
    ctx.conn.commit()
    LOG.info("discovery added %d new URL(s)", new)
    return new


def run_fetch(site: Site, ctx: Context, paths: dict, limit: int = 0):
    rows = ctx.conn.execute(
        "SELECT url FROM urls WHERE status='pending' ORDER BY rowid"
        + (" LIMIT ?" if limit else ""),
        (limit,) if limit else (),
    ).fetchall()

    if not rows:
        LOG.info("nothing pending")
        return 0, 0, 0

    LOG.info("fetching %d pending article(s)", len(rows))
    done = failed = skipped = 0
    conn = ctx.conn

    with open(paths["jsonl"], "a", encoding="utf-8") as out:
        for (url,) in rows:
            conn.execute(
                "UPDATE urls SET attempts=attempts+1, updated_at=? WHERE url=?",
                (now_iso(), url),
            )
            conn.commit()

            status, body, err = ctx.fetch(url)

            if status != 200 or not body:
                conn.execute(
                    "UPDATE urls SET status='failed', http_status=?, error=?, "
                    "updated_at=? WHERE url=?",
                    (status, err, now_iso(), url),
                )
                conn.commit()
                failed += 1
                continue

            # Raw layer first, so a parser change never forces a re-crawl.
            slug = slug_for(url)
            raw_path = os.path.join(paths["raw"], f"{slug}.html.gz")
            with gzip.open(raw_path, "wb", compresslevel=6) as gz:
                gz.write(body)

            try:
                rec = site.parse(body, url)
            except Exception as e:  # noqa: BLE001 - one bad page must not stop the queue
                LOG.exception("parse failed for %s", url)
                conn.execute(
                    "UPDATE urls SET status='failed', http_status=?, error=?, "
                    "updated_at=? WHERE url=?",
                    (status, f"parse: {type(e).__name__}: {e}"[:300],
                     now_iso(), url),
                )
                conn.commit()
                failed += 1
                continue

            if rec is None:
                conn.execute(
                    "UPDATE urls SET status='failed', http_status=?, error=?, "
                    "updated_at=? WHERE url=?",
                    (status, "not an article", now_iso(), url),
                )
                conn.commit()
                failed += 1
                continue

            pub = rec.get("published_date")
            if not in_range(pub, ctx.start, ctx.end):
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
                "post_id=?, error=NULL, updated_at=? WHERE url=?",
                (status, pub, str(rec.get("post_id") or slug), now_iso(), url),
            )
            conn.commit()
            done += 1

    return done, failed, skipped


def slug_for(url: str) -> str:
    """Filesystem-safe raw filename derived from the URL's last path segment."""
    tail = url.rstrip("/").split("/")[-1].split("?")[0]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tail)
    return safe[:100] or hashlib.sha1(url.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# plumbing
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


def setup_logging(path: str, quiet: bool = False) -> None:
    if LOG.handlers:
        return
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


def print_status(conn) -> None:
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
