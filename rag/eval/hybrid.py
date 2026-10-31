#!/usr/bin/env python3
"""Hybrid retrieval: dense BGE-M3 score + a lexical channel for place names.

Why a lexical channel at all: BGE-M3 aligns *translated* entity names well
(NUG 0.790) but barely aligns *transliterated* ones -- place names 0.487,
person names 0.443, with `Mandalay` closer to `နေပြည်တော်` than to `မန္တလေး`.
Every known-item miss hinged on a transliterated name, so the dense score
alone cannot tell Ramree from Myingyan.

How it works: the query's English place names are looked up in a gazetteer
(built by mining the corpus, not by guessing spellings), and a document that
contains the corresponding Burmese form gets a bonus.

Two details that are easy to get wrong:

  - Match forms carry an administrative suffix (ဆောမြို့, not ဆော). Burmese
    has no word boundaries, and bare short names are substrings of common
    words: `ဆော` alone matches 117,574 times, `ဆောမြို့` only 418.
  - Documents are whitespace-stripped before matching, because Burmese spaces
    words inconsistently -- the same entity appears several ways.

alpha is chosen leave-one-out: with only 29 questions, picking one alpha on
all of them and reporting the result would be fitting noise.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import embed  # noqa: E402

HERE = Path(__file__).resolve().parent
IDX = HERE.parents[0] / "index"
GAZ = HERE.parents[0] / "gazetteer" / "places.json"
WS = re.compile(r"\s+")


def norm(t: str) -> str:
    """Whitespace-stripped AND NFC-normalised.

    18% of the indexed articles are not NFC. Burmese combining marks can be
    stored in either order and render identically, so a substring match
    against a differently-ordered form silently misses. Normalising both
    sides recovers 34 (article, place) pairs across the gazetteer.
    """
    return WS.sub("", unicodedata.normalize("NFC", t))


def load_bodies(urls: set[str]) -> dict[str, str]:
    """Whitespace-stripped text for the indexed documents."""
    out = {}
    import glob
    for f in sorted(glob.glob("data-vm/*/parsed/articles.jsonl")):
        for line in open(f):
            d = json.loads(line)
            if d["url"] in urls:
                out[d["url"]] = norm((d.get("title") or "") + (d.get("body") or ""))
    return out


def main() -> None:
    gaz = json.load(open(GAZ))
    qs = [json.loads(l) for l in (HERE / "questions.jsonl").open()]
    ki = [q for q in qs if q["type"] == "known_item"]

    vecs = np.load(IDX / "vectors.npy")
    meta = [json.loads(l) for l in (IDX / "meta.jsonl").open()]
    urls = [m["url"] for m in meta]
    row = {u: i for i, u in enumerate(urls)}
    bodies = load_bodies(set(urls))
    print(f"index {len(vecs)} docs, bodies loaded for {len(bodies)}")

    qv = embed.encode([q["question"] for q in ki], max_len=64, batch=16)
    dense = vecs @ qv.T                                  # (ndocs, nq)

    # lexical channel ---------------------------------------------------
    lex = np.zeros_like(dense)
    found = []
    for j, q in enumerate(ki):
        hits = [en for en in gaz if re.search(rf"\b{re.escape(en)}\b", q["question"], re.I)]
        found.append(hits)
        if not hits:
            continue
        forms = [norm(gaz[en]["match"]) for en in hits]
        for i, u in enumerate(urls):
            b = bodies.get(u, "")
            if any(f in b for f in forms):
                lex[i, j] = 1.0

    n_sig = sum(1 for h in found if h)
    print(f"queries with a gazetteer hit: {n_sig}/{len(ki)}")
    for q, h in zip(ki, found):
        if h:
            print(f"   {q['qid']}: {', '.join(h)}  -> {lex[:, ki.index(q)].sum():.0f} docs match")

    def ranks_for(alpha: float) -> list[int | None]:
        out = []
        for j, q in enumerate(ki):
            s = dense[:, j] + alpha * lex[:, j]
            order = np.argsort(-s)
            tgt = set(q["target_urls"])
            r = next((k + 1 for k, i in enumerate(order[:50]) if urls[i] in tgt), None)
            out.append(r)
        return out

    GRID = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
    print("\n%-7s %8s %8s %8s %8s" % ("alpha", "R@1", "R@5", "R@10", "MRR"))
    all_r = {}
    for a in GRID:
        r = ranks_for(a)
        all_r[a] = r
        n = len(r)
        f = lambda k: sum(1 for x in r if x and x <= k) / n
        mrr = sum(1 / x for x in r if x) / n
        print("%-7.2f %8.3f %8.3f %8.3f %8.3f" % (a, f(1), f(5), f(10), mrr))

    # leave-one-out: alpha chosen on the other 28, scored on the held-out one
    loo = []
    for h in range(len(ki)):
        best, best_s = 0.0, -1.0
        for a in GRID:
            r = all_r[a]
            s = sum(1 / x for i, x in enumerate(r) if x and i != h) / (len(r) - 1)
            if s > best_s:
                best_s, best = s, a
        loo.append(all_r[best][h])
    n = len(loo)
    f = lambda k: sum(1 for x in loo if x and x <= k) / n
    print("\nLEAVE-ONE-OUT (honest: alpha never saw the question it scores)")
    print("  R@1 %.3f   R@5 %.3f   R@10 %.3f   MRR %.3f"
          % (f(1), f(5), f(10), sum(1 / x for x in loo if x) / n))
    print("  baseline alpha=0: R@1 %.3f  R@5 %.3f  R@10 %.3f  MRR %.3f"
          % (sum(1 for x in all_r[0.0] if x and x <= 1) / n,
             sum(1 for x in all_r[0.0] if x and x <= 5) / n,
             sum(1 for x in all_r[0.0] if x and x <= 10) / n,
             sum(1 / x for x in all_r[0.0] if x) / n))


if __name__ == "__main__":
    main()
