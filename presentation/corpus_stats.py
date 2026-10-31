#!/usr/bin/env python3
"""Recompute the preprocessing funnel from data-vm/, replicating the exact
filter order in classifier/build_dataset.py. Writes presentation/corpus_stats.json.

Read-only on all data directories.
"""
from __future__ import annotations
import collections, json, os, sys, unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from classifier.build_dataset import SITES, MIN_CHARS, norm_text  # same constants

DATA = os.path.join(ROOT, "data-vm")
OUT = os.path.join(ROOT, "presentation", "corpus_stats.json")

def main() -> int:
    per_site = {s: collections.Counter() for s in SITES}
    months = collections.Counter()
    months_by_site = collections.defaultdict(collections.Counter)
    lengths = []                      # char length of KEPT bodies
    lengths_by_site = collections.defaultdict(list)
    seen_urls: set[str] = set()
    seen_bodies: set[str] = set()
    global_stats = collections.Counter()

    for site in SITES:
        path = os.path.join(DATA, site, "parsed", "articles.jsonl")
        if not os.path.exists(path):
            print(f"MISSING {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                per_site[site]["read"] += 1
                global_stats["read"] += 1
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    per_site[site]["bad_json"] += 1
                    global_stats["bad_json"] += 1
                    continue

                url = r.get("url")
                if not url or url in seen_urls:
                    per_site[site]["dup_url"] += 1
                    global_stats["dup_url"] += 1
                    continue
                seen_urls.add(url)

                body = norm_text(r.get("body"))
                n = len(body.strip())
                if n < MIN_CHARS:
                    per_site[site]["too_short"] += 1
                    global_stats["too_short"] += 1
                    continue

                h = r.get("content_sha256")
                key = h or body[:400]
                if key in seen_bodies:
                    per_site[site]["dup_body"] += 1
                    global_stats["dup_body"] += 1
                    continue
                seen_bodies.add(key)

                per_site[site]["kept"] += 1
                global_stats["kept"] += 1
                lengths.append(n)
                lengths_by_site[site].append(n)
                d = (r.get("published_date") or "")[:7]
                if len(d) == 7 and d[4] == "-":
                    months[d] += 1
                    months_by_site[site][d] += 1
                else:
                    global_stats["no_date"] += 1

    out = {
        "source": "data-vm/<site>/parsed/articles.jsonl",
        "filters": {"MIN_CHARS": MIN_CHARS, "order":
                    ["read", "bad_json", "dup_url", "too_short", "dup_body", "kept"]},
        "global": dict(global_stats),
        "per_site": {s: dict(c) for s, c in per_site.items()},
        "months": dict(sorted(months.items())),
        "months_by_site": {s: dict(sorted(c.items())) for s, c in months_by_site.items()},
        "length_hist": None,
        "lengths_summary": {},
    }

    import statistics as st
    def summ(v):
        v = sorted(v)
        return {"n": len(v), "mean": st.mean(v), "median": st.median(v),
                "p10": v[int(.10 * len(v))], "p90": v[int(.90 * len(v))],
                "min": v[0], "max": v[-1]}
    out["lengths_summary"]["all"] = summ(lengths)
    for s, v in lengths_by_site.items():
        out["lengths_summary"][s] = summ(v)
    # coarse histogram of kept body lengths, 200-char bins up to 10k
    hist = collections.Counter(min(int(n // 200) * 200, 10000) for n in lengths)
    out["length_hist"] = dict(sorted(hist.items()))

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["global"], indent=1))
    print("\nper site:")
    for s in SITES:
        c = per_site[s]
        print(f"  {s:12s} read {c['read']:6d}  too_short {c['too_short']:5d}  "
              f"dup_url {c['dup_url']:5d}  dup_body {c['dup_body']:5d}  kept {c['kept']:6d}")
    print(f"\nmonths: {min(months)} .. {max(months)}  ({len(months)} months)")
    print(f"wrote {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
