#!/usr/bin/env python3
"""Compare lexical-bonus weighting schemes for hybrid retrieval.

The shipped channel adds a flat +alpha to any document containing the Burmese
form of a place named in the query. That is provably wasted on frequent terms:
`Thailand` matches 19% of the corpus and moves no question at any alpha, while
`Ramree` (0.1%) took its question from miss to rank 21.

Weighting the bonus by inverse document frequency is the standard fix and --
unlike per-entity suffix tuning -- introduces no parameter chosen on the test
questions. idf is a property of the corpus, not of the 29 questions.

Schemes:
    flat      alpha * 1[any form matches]
    idf-max   alpha * max idf over the query's matched forms present in the doc
    idf-sum   alpha * sum idf over them

alpha is always selected leave-one-out: with n=29, choosing it on all 29 and
reporting the result is fitting noise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import embed  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]


def main() -> None:
    cache = np.load(ROOT / ".scratch/lexcache.npz")
    M, df = cache["M"], cache["df"]
    keys = json.load(open(ROOT / ".scratch/lexcache_keys.json"))
    N = M.shape[0]
    idf = np.log(N / np.maximum(df, 1))

    urls = [json.loads(l)["url"] for l in open(ROOT / "rag/index/meta.jsonl")]
    vecs = np.load(ROOT / "rag/index/vectors.npy")
    qs = [json.loads(l) for l in (HERE / "questions.jsonl").open()]
    ki = [q for q in qs if q["type"] == "known_item"]

    qv = embed.encode([q["question"] for q in ki], max_len=64, batch=16)
    dense = vecs @ qv.T

    # which gazetteer forms does each question mention?
    matched = []
    for q in ki:
        cols = [j for j, k in enumerate(keys)
                if re.search(rf"\b{re.escape(k['english'])}\b", q["question"], re.I)]
        matched.append(cols)

    def bonus(scheme: str) -> np.ndarray:
        B = np.zeros_like(dense)
        for j, cols in enumerate(matched):
            if not cols:
                continue
            sub = M[:, cols]                      # (ndocs, nforms)
            if scheme == "flat":
                B[:, j] = sub.any(1).astype(float)
            elif scheme == "idf-max":
                B[:, j] = (sub * idf[cols]).max(1)
            elif scheme == "idf-sum":
                B[:, j] = (sub * idf[cols]).sum(1)
        return B

    def ranks(B: np.ndarray, a: float) -> list[int | None]:
        out = []
        for j, q in enumerate(ki):
            s = dense[:, j] + a * B[:, j]
            order = np.argsort(-s)
            tgt = set(q["target_urls"])
            out.append(next((k + 1 for k, i in enumerate(order[:300])
                             if urls[i] in tgt), None))
        return out

    def stats(r):
        n = len(r)
        f = lambda k: sum(1 for x in r if x and x <= k) / n
        return f(1), f(5), f(10), sum(1 / x for x in r if x) / n

    print("questions with >=1 gazetteer form: %d/%d"
          % (sum(1 for c in matched if c), len(ki)))
    print("\n%-22s %6s %6s %6s %6s" % ("scheme", "R@1", "R@5", "R@10", "MRR"))
    print("-" * 56)
    print("%-22s %6.3f %6.3f %6.3f %6.3f" % ("dense only", *stats(ranks(np.zeros_like(dense), 0.0))))

    results = {}
    for scheme in ("flat", "idf-max", "idf-sum"):
        B = bonus(scheme)
        allr = {a: ranks(B, a) for a in GRID}
        results[scheme] = allr
        best = max(GRID, key=lambda a: stats(allr[a])[3])
        print("%-22s %6.3f %6.3f %6.3f %6.3f   (best alpha=%.2f, tuned on all 29)"
              % (scheme, *stats(allr[best]), best))

    print("\nLEAVE-ONE-OUT  (alpha picked on the other 28 by MRR)")
    print("%-22s %6s %6s %6s %6s" % ("scheme", "R@1", "R@5", "R@10", "MRR"))
    print("-" * 56)
    loo_ranks = {}
    for scheme, allr in results.items():
        loo = []
        for h in range(len(ki)):
            best, bs = 0.0, -1.0
            for a in GRID:
                r = allr[a]
                s = sum(1 / x for i, x in enumerate(r) if x and i != h) / (len(r) - 1)
                if s > bs:
                    bs, best = s, a
            loo.append(allr[best][h])
        loo_ranks[scheme] = loo
        print("%-22s %6.3f %6.3f %6.3f %6.3f" % (scheme, *stats(loo)))

    base = ranks(np.zeros_like(dense), 0.0)
    print("\nPER-QUESTION (leave-one-out ranks)")
    print("%-5s %8s %8s %8s %8s  %s" % ("qid", "dense", "flat", "idf-max", "idf-sum", "forms"))
    print("-" * 78)
    f = lambda x: "miss" if x is None else str(x)
    for j, q in enumerate(ki):
        names = ",".join(keys[c]["english"] for c in matched[j]) or "-"
        print("%-5s %8s %8s %8s %8s  %s"
              % (q["qid"], f(base[j]), f(loo_ranks["flat"][j]),
                 f(loo_ranks["idf-max"][j]), f(loo_ranks["idf-sum"][j]), names))

    for scheme in ("flat", "idf-max", "idf-sum"):
        imp = wor = same = 0
        for b, a in zip(base, loo_ranks[scheme]):
            if b is None and a is None: same += 1
            elif b is None: imp += 1
            elif a is None: wor += 1
            elif a < b: imp += 1
            elif a > b: wor += 1
            else: same += 1
        print("%-10s vs dense:  improved %d  worsened %d  unchanged %d"
              % (scheme, imp, wor, same))


if __name__ == "__main__":
    main()
