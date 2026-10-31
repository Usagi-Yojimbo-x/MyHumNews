#!/usr/bin/env python3
"""
Turn the five-outlet corpus into a labelled topic-classification dataset.

    python -m classifier.build_dataset --out classifier/data

Pipeline, in order, with every exclusion counted so nothing vanishes silently:

  1. load all five outlets' JSONL
  2. drop duplicate URLs            (restart artefacts -- DVB has 5)
  3. drop duplicate bodies          (RFA's 117 programme-announcement stubs)
  4. drop very short bodies         (<MIN_CHARS: no text to classify)
  5. map outlet labels -> topic     (classifier.taxonomy)
  6. drop articles with no topic    (format/placement-only labels)
  7. split train/test

TEXT NORMALISATION: bodies are NFC-normalised for features only. Burmese
combining marks appear in either order and render identically, so without this
the same word produces different character n-grams -- the same trap that
silently broke the label mapping. The raw layer and the JSONL on disk are left
exactly as crawled; normalisation happens here, downstream.

SPLIT: temporal by default (train on earlier dates, test on later). A random
split leaks badly on news -- five outlets covering the same event on the same
day put near-identical text in both halves, which inflates accuracy without
measuring generalisation. Use --split random to see that effect for yourself;
expect it to look better and mean less.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier.taxonomy import fingerprint, label_article  # noqa: E402

SITES = ["dvb", "bbc", "rfa", "irrawaddy", "myanmarnow"]
MIN_CHARS = 200


def norm_text(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def iter_articles(data_root: str):
    for site in SITES:
        path = os.path.join(data_root, site, "parsed", "articles.jsonl")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield site, json.loads(line)
                except json.JSONDecodeError:
                    yield site, None


def build(data_root: str, out_dir: str, split: str, test_frac: float, seed: int):
    stats = collections.Counter()
    seen_urls: set[str] = set()
    seen_bodies: set[str] = set()
    rows = []

    for site, r in iter_articles(data_root):
        stats["read"] += 1
        if r is None:
            stats["bad_json"] += 1
            continue

        url = r.get("url")
        if not url or url in seen_urls:
            stats["dup_url"] += 1
            continue
        seen_urls.add(url)

        body = norm_text(r.get("body"))
        if len(body.strip()) < MIN_CHARS:
            stats["too_short"] += 1
            continue

        # Exact-body dedup, within and across outlets. Catches crawl artefacts
        # and repeated boilerplate; will not catch near-duplicates.
        h = r.get("content_sha256")
        key = h or body[:400]
        if key in seen_bodies:
            stats["dup_body"] += 1
            continue
        seen_bodies.add(key)

        labels = r.get("categories") or ([r["category"]] if r.get("category") else [])
        topic = label_article(labels)
        if topic is None:
            stats["no_topic"] += 1
            continue

        rows.append({
            "url": url,
            "outlet": r.get("outlet") or site,
            "site": site,
            "title": norm_text(r.get("title")),
            "text": body,
            "published_date": r.get("published_date"),
            "topic": topic,
            "raw_labels": labels,
        })
        stats["kept"] += 1

    # ---- split ----
    if split == "temporal":
        rows.sort(key=lambda r: r["published_date"] or "")
        cut = int(len(rows) * (1 - test_frac))
        train, test = rows[:cut], rows[cut:]
        boundary = test[0]["published_date"][:10] if test else None
    else:
        random.Random(seed).shuffle(rows)
        cut = int(len(rows) * (1 - test_frac))
        train, test = rows[:cut], rows[cut:]
        boundary = None

    os.makedirs(out_dir, exist_ok=True)

    # Stamp the taxonomy that produced these labels. Consumers check it, so a
    # taxonomy edit can no longer leave a stale dataset in place unnoticed.
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "taxonomy_fingerprint": fingerprint(),
            "corpus_root": os.path.abspath(data_root),
            "split": split, "test_frac": test_frac, "seed": seed,
            "min_chars": MIN_CHARS,
            "n_train": len(train), "n_test": len(test),
            "classes": sorted({r["topic"] for r in train}),
            "built_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        }, fh, indent=2, sort_keys=True)

    for name, part in (("train", train), ("test", test)):
        p = os.path.join(out_dir, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- report ----
    print("=== pipeline ===")
    order = ["read", "bad_json", "dup_url", "too_short", "dup_body",
             "no_topic", "kept"]
    for k in order:
        if k in stats:
            print(f"  {k:10} {stats[k]:>7}")
    if stats["read"]:
        print(f"  kept {stats['kept'] / stats['read'] * 100:.1f}% of articles read")

    print(f"\n=== split: {split} ===")
    print(f"  train {len(train)}   test {len(test)}")
    if boundary:
        print(f"  temporal boundary: test is everything from {boundary} onward")

    print("\n=== class distribution ===")
    tr = collections.Counter(r["topic"] for r in train)
    te = collections.Counter(r["topic"] for r in test)
    print(f"  {'topic':14} {'train':>7} {'test':>7}")
    for c, n in tr.most_common():
        print(f"  {c:14} {n:>7} {te.get(c, 0):>7}")
    missing = set(tr) - set(te)
    if missing:
        print(f"  WARNING: classes absent from test: {sorted(missing)}")

    print("\n=== topic x outlet (train) ===")
    by = collections.defaultdict(collections.Counter)
    for r in train:
        by[r["topic"]][r["site"]] += 1
    hdr = f"  {'topic':14}" + "".join(f"{s[:9]:>10}" for s in SITES)
    print(hdr)
    for c, _ in tr.most_common():
        print(f"  {c:14}" + "".join(f"{by[c].get(s, 0):>10}" for s in SITES))

    print(f"\nwrote {out_dir}/train.jsonl, test.jsonl and manifest.json")
    print(f"taxonomy fingerprint {fingerprint()}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data", help="corpus root (default: data)")
    p.add_argument("--out", default="classifier/data")
    p.add_argument("--split", choices=["temporal", "random"], default="temporal")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()
    build(a.data, a.out, a.split, a.test_frac, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
