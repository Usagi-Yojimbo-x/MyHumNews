#!/usr/bin/env python3
"""
Corpus verification for MyHumNews.

    python verify.py                # every outlet under data/
    python verify.py dvb bbc        # named outlets

Streams the JSONL (never loads a corpus into memory) and reports integrity,
coverage and anything that would undermine a cross-outlet comparison.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys

FIELDS = ["url", "outlet", "title", "body", "published_date", "category",
          "scraped_at", "content_sha256"]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def verify(site: str) -> dict | None:
    path = f"data/{site}/parsed/articles.jsonl"
    if not os.path.exists(path):
        print(f"  {site}: no JSONL at {path}")
        return None

    n = 0
    urls, hashes = set(), collections.Counter()
    months = collections.Counter()
    cats = collections.Counter()
    empty = collections.Counter()
    bad_json = 0
    missing_field = collections.Counter()
    body_chars = 0
    dates: list[str] = []

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            n += 1
            for f in FIELDS:
                if f not in r:
                    missing_field[f] += 1
            urls.add(r.get("url"))
            h = r.get("content_sha256")
            if h:
                hashes[h] += 1
            pub = r.get("published_date") or ""
            if pub:
                months[pub[:7]] += 1
                dates.append(pub[:10])
            for f in ("title", "body", "published_date"):
                if not (r.get(f) or "").strip():
                    empty[f] += 1
            body_chars += len(r.get("body") or "")
            c = r.get("category")
            if c:
                cats[c] += 1

    raw_dir = f"data/{site}/raw"
    raw_n = len(glob.glob(f"{raw_dir}/*.gz")) if os.path.isdir(raw_dir) else 0
    raw_bytes = sum(
        os.path.getsize(p) for p in glob.glob(f"{raw_dir}/*.gz")
    ) if raw_n else 0

    dup_hashes = sum(c - 1 for c in hashes.values() if c > 1)

    print(f"\n=== {site} ===")
    print(f"  records            {n:,}")
    print(f"  unique urls        {len(urls):,}"
          + ("  <-- DUPLICATE LINES" if len(urls) != n else ""))
    print(f"  raw files          {raw_n:,} ({human(raw_bytes)})")
    print(f"  jsonl size         {human(os.path.getsize(path))}")
    if dates:
        print(f"  date range         {min(dates)} .. {max(dates)}")
    print(f"  body text          {body_chars:,} chars "
          f"(mean {body_chars // max(n, 1):,}/article)")
    print(f"  distinct categories {len(cats)}")

    problems = []
    if bad_json:
        problems.append(f"{bad_json} unparseable lines")
    if len(urls) != n:
        problems.append(f"{n - len(urls)} duplicate URLs")
    for f, c in empty.items():
        problems.append(f"{c} empty {f}")
    for f, c in missing_field.items():
        problems.append(f"{c} missing field '{f}'")
    if dup_hashes:
        problems.append(f"{dup_hashes} repeated content_sha256 (near-dupe bodies)")

    print("  integrity          " + ("OK" if not problems else "; ".join(problems)))

    if months:
        print("  per-month coverage:")
        lo = min(months)
        hi = max(months)
        peak = max(months.values())
        for m in sorted(months):
            bar = "#" * max(1, round(28 * months[m] / peak))
            print(f"    {m}  {months[m]:>5}  {bar}")
        # A gap is a month with no articles at all between first and last.
        span = set()
        y, mo = int(lo[:4]), int(lo[5:7])
        while f"{y:04d}-{mo:02d}" <= hi:
            span.add(f"{y:04d}-{mo:02d}")
            mo += 1
            if mo == 13:
                y, mo = y + 1, 1
        gaps = sorted(span - set(months))
        if gaps:
            print(f"  MISSING MONTHS     {', '.join(gaps)}")

    return {"site": site, "n": n, "hashes": set(hashes), "months": months}


def main() -> int:
    sites = sys.argv[1:] or sorted(
        d for d in os.listdir("data")
        if os.path.isdir(f"data/{d}")
    ) if os.path.isdir("data") else []
    if not sites:
        print("no outlets found under data/")
        return 1

    results = [r for r in (verify(s) for s in sites) if r]

    if len(results) > 1:
        print("\n=== cross-outlet ===")
        total = sum(r["n"] for r in results)
        print(f"  corpus total       {total:,} articles")
        for a in range(len(results)):
            for b in range(a + 1, len(results)):
                shared = results[a]["hashes"] & results[b]["hashes"]
                if shared:
                    print(f"  {results[a]['site']} / {results[b]['site']}: "
                          f"{len(shared)} identical bodies (syndicated)")
        print("  monthly volume by outlet:")
        allm = sorted({m for r in results for m in r["months"]})
        hdr = "  " + " " * 9 + "".join(f"{r['site']:>8}" for r in results)
        print(hdr)
        for m in allm:
            row = "".join(f"{r['months'].get(m, 0):>8}" for r in results)
            print(f"    {m} {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
