#!/usr/bin/env python3
"""Score the chunked index against the whole-article baseline.

    .venv/bin/python rag/eval/eval_chunked.py

Both indexes cover the SAME 38,092 articles from data-vm, so pooling is the
only thing that differs. A document's score is the max over its chunks.

This answers the question the earlier probe could not. That probe compared
each target's whole-article score against its best chunk and found large gains
where the answer sits deep in a long body (K21 +0.176, K22 +0.130, K17 +0.076)
and nothing on the geography misses (~+0.004). But max-over-chunks lifts every
document's score, distractors included, so a score gain need not be a rank
gain. Only ranks over the full chunked index settle it.

Also reports the hybrid combination, since the gazetteer channel is orthogonal
to pooling and the two could stack.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import embed  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]


def stats(r):
    n = len(r)
    f = lambda k: sum(1 for x in r if x and x <= k) / n
    return f(1), f(5), f(10), sum(1 / x for x in r if x) / n


def main() -> None:
    ck = ROOT / "rag/index_chunked"
    if not (ck / "vectors.npy").exists():
        sys.exit("chunked index not built yet: run rag.build_index_chunked first")

    qs = [json.loads(l) for l in (ROOT / "rag/eval/questions.jsonl").open()]
    ki = [q for q in qs if q["type"] == "known_item"]
    qv = embed.encode([q["question"] for q in ki], max_len=64, batch=16)

    # ---- whole-article baseline --------------------------------------
    wv = np.load(ROOT / "rag/index/vectors.npy")
    wurls = [json.loads(l)["url"] for l in (ROOT / "rag/index/meta.jsonl").open()]
    wdense = wv @ qv.T

    # ---- chunked: max over each document's chunks --------------------
    cv = np.load(ck / "vectors.npy")
    cmeta = [json.loads(l) for l in (ck / "meta.jsonl").open()]
    n = min(len(cv), len(cmeta))
    cv, cmeta = cv[:n], cmeta[:n]          # tolerate a checkpoint mid-write
    csc = cv @ qv.T
    by = defaultdict(list)
    for i, m in enumerate(cmeta):
        by[m["url"]].append(i)
    curls = list(by)
    cdense = np.zeros((len(curls), len(ki)), dtype=np.float32)
    for d, u in enumerate(curls):
        cdense[d] = csc[by[u]].max(0)
    print(f"whole-article: {len(wurls)} docs   chunked: {len(curls)} docs, "
          f"{len(cmeta)} chunks ({len(cmeta)/max(len(curls),1):.2f} per doc)")

    def ranks(dense, urls, B=None, a=0.0):
        out = []
        for j, q in enumerate(ki):
            s = dense[:, j] + (a * B[:, j] if B is not None else 0.0)
            order = np.argsort(-s)
            tgt = set(q["target_urls"])
            out.append(next((k + 1 for k, i in enumerate(order[:300])
                             if urls[i] in tgt), None))
        return out

    wr = ranks(wdense, wurls)
    cr = ranks(cdense, curls)
    print("\n%-30s %6s %6s %6s %6s" % ("pooling", "R@1", "R@5", "R@10", "MRR"))
    print("-" * 62)
    print("%-30s %6.3f %6.3f %6.3f %6.3f" % ("whole article (512 tok)", *stats(wr)))
    print("%-30s %6.3f %6.3f %6.3f %6.3f" % ("chunked, max-over-chunks", *stats(cr)))

    # ---- hybrid on top of each ---------------------------------------
    cache = np.load(ROOT / ".scratch/lexcache.npz")
    keys = json.load(open(ROOT / ".scratch/lexcache_keys.json"))
    M = cache["M"]
    wrow = {u: i for i, u in enumerate(wurls)}
    Bw = np.zeros_like(wdense)
    Bc = np.zeros((len(curls), len(ki)), dtype=np.float32)
    for j, q in enumerate(ki):
        cols = [x for x, k in enumerate(keys)
                if re.search(rf"\b{re.escape(k['english'])}\b", q["question"], re.I)]
        if not cols:
            continue
        hit = M[:, cols].any(1)
        Bw[:, j] = hit.astype(float)
        for d, u in enumerate(curls):
            i = wrow.get(u)
            if i is not None and hit[i]:
                Bc[d, j] = 1.0

    def loo(dense, urls, B):
        allr = {a: ranks(dense, urls, B, a) for a in GRID}
        out = []
        for h in range(len(ki)):
            best, bs = 0.0, -1.0
            for a in GRID:
                r = allr[a]
                s = sum(1 / x for i, x in enumerate(r) if x and i != h) / (len(r) - 1)
                if s > bs:
                    bs, best = s, a
            out.append(allr[best][h])
        return out

    hw, hc = loo(wdense, wurls, Bw), loo(cdense, curls, Bc)
    print("%-30s %6.3f %6.3f %6.3f %6.3f" % ("whole + gazetteer (LOO)", *stats(hw)))
    print("%-30s %6.3f %6.3f %6.3f %6.3f" % ("chunked + gazetteer (LOO)", *stats(hc)))

    print("\nper-question")
    print("%-5s %8s %8s %10s %10s" % ("qid", "whole", "chunked", "whole+gaz", "chunk+gaz"))
    f = lambda x: "miss" if x is None else str(x)
    for j, q in enumerate(ki):
        print("%-5s %8s %8s %10s %10s"
              % (q["qid"], f(wr[j]), f(cr[j]), f(hw[j]), f(hc[j])))

    for nm, r in (("chunked vs whole", cr), ("chunk+gaz vs whole", hc)):
        imp = wor = same = 0
        for x, y in zip(wr, r):
            if x is None and y is None: same += 1
            elif x is None: imp += 1
            elif y is None: wor += 1
            elif y < x: imp += 1
            elif y > x: wor += 1
            else: same += 1
        print("%-22s improved %d  worsened %d  unchanged %d" % (nm, imp, wor, same))


if __name__ == "__main__":
    main()
