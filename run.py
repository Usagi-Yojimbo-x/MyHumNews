#!/usr/bin/env python3
"""
MyHumNews corpus crawler -- entry point.

    python run.py <site> <phase> [options]

    site    dvb | bbc        (see crawler/sites/)
    phase   discover | fetch | run | status

Examples:
    python run.py bbc run --limit 20        # 20-article smoke test
    python run.py dvb run                   # full crawl, Jan 2025 onward
    python run.py bbc status
"""

from __future__ import annotations

import argparse
import importlib
import sys

from crawler import core

SITES = ["dvb", "bbc", "rfa", "irrawaddy", "myanmarnow"]

# Corpus-wide defaults; per-site politeness comes from the adapter's `delay`.
START_DATE = "2025-01-01"
END_DATE = None
OUTPUT_ROOT = "data"


def load_site(slug: str):
    try:
        return importlib.import_module(f"crawler.sites.{slug}")
    except ModuleNotFoundError as e:
        raise SystemExit(f"unknown site '{slug}' ({e}); available: {', '.join(SITES)}")


def main() -> int:
    p = argparse.ArgumentParser(description="MyHumNews Burmese news crawler")
    p.add_argument("site", help=f"one of: {', '.join(SITES)}")
    p.add_argument("phase", choices=["discover", "fetch", "run", "status"])
    p.add_argument("--out", default=None, help="default: data/<site>")
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=END_DATE)
    p.add_argument("--delay", type=float, default=None,
                   help="override the site's politeness delay")
    p.add_argument("--limit", type=int, default=0,
                   help="cap URLs discovered / articles fetched (testing)")
    p.add_argument("--max-pages", type=int, default=0,
                   help="cap listing pages per category/topic (testing)")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    site = load_site(args.site)
    outdir = args.out or f"{OUTPUT_ROOT}/{site.name}"
    paths = core.setup_paths(outdir)
    core.setup_logging(paths["log"], args.quiet)
    conn = core.open_db(paths["db"])

    if args.phase == "status":
        core.print_status(conn)
        return 0

    delay = args.delay if args.delay is not None else site.delay
    # A site may ask for the curl transport (see CurlFetcher's docstring).
    fetcher_cls = (core.CurlFetcher
                   if getattr(site, "transport", "requests") == "curl"
                   else core.Fetcher)
    fetcher = fetcher_cls(delay, core.MAX_ATTEMPTS)
    ctx = core.Context(conn, fetcher, args.start, args.end, args.max_pages)

    core.LOG.info("=== %s %s | out=%s delay=%.1fs range=%s..%s ===",
                  site.name, args.phase, outdir, delay,
                  args.start, args.end or "today")

    if args.retry_failed:
        n = conn.execute(
            "UPDATE urls SET status='pending', attempts=0 WHERE status='failed'"
        ).rowcount
        conn.commit()
        core.LOG.info("reset %d failed URL(s) to pending", n)

    if args.phase in ("discover", "run"):
        core.run_discover(site, ctx, args.limit)

    if args.phase in ("fetch", "run"):
        done, failed, skipped = core.run_fetch(site, ctx, paths, args.limit)
        core.LOG.info("fetch complete: %d done, %d failed, %d skipped",
                      done, failed, skipped)

    core.print_status(conn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
