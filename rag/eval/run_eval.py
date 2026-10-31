#!/usr/bin/env python3
"""Score the question set against the BGE-M3 index.

known_item : the answer is one named article, so Recall@k and MRR are
             computable with no human judgement at all.
negative   : no article answers it. There is no right answer to retrieve,
             so the only thing to report is the top similarity score --
             if negatives score as high as known-items, a score threshold
             cannot tell "no answer" from "answer".
topical    : many articles are relevant and nobody has said which. This
             script only dumps the top-k for a human to judge later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import embed  # noqa: E402

HERE = Path(__file__).resolve().parent
IDX = HERE.parents[0] / "index"


def load_index():
    vecs = np.load(IDX / "vectors.npy")
    meta = [json.loads(l) for l in (IDX / "meta.jsonl").open()]
    assert len(meta) == len(vecs), f"{len(meta)} meta vs {len(vecs)} vectors"
    return vecs, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(HERE / "questions.jsonl"))
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--dump", help="write top-k rows here as jsonl")
    args = ap.parse_args()

    qs = [json.loads(l) for l in Path(args.questions).open()]
    vecs, meta = load_index()
    url_row = {m["url"]: i for i, m in enumerate(meta)}
    print(f"index: {len(vecs)} articles, dim {vecs.shape[1]}")

    # Coverage first: a target that is not in the index can never be found,
    # and counting it as a miss would blame the retriever for the indexer.
    ki = [q for q in qs if q["type"] == "known_item"]
    missing = [(q["qid"], u) for q in ki for u in q["target_urls"]
               if u not in url_row]
    if missing:
        print(f"\n!! {len(missing)} target urls are NOT in the index:")
        for qid, u in missing:
            print(f"   {qid}  {u}")

    qvecs = embed.encode([q["question"] for q in qs], max_len=512, batch=16)
    sims = vecs @ qvecs.T                       # (n_docs, n_q)

    dump = open(args.dump, "w") if args.dump else None
    ranks: list[int | None] = []
    rows_by_type: dict[str, list] = {}

    for j, q in enumerate(qs):
        col = sims[:, j]
        top = np.argsort(-col)[:args.k]
        hits = [{"rank": r + 1, "url": meta[i]["url"], "outlet": meta[i]["outlet"],
                 "title": meta[i]["title"], "score": round(float(col[i]), 4)}
                for r, i in enumerate(top)]
        if dump:
            dump.write(json.dumps({"qid": q["qid"], "type": q["type"],
                                   "question": q["question"], "hits": hits},
                                  ensure_ascii=False) + "\n")
        rows_by_type.setdefault(q["type"], []).append((q, hits))

        if q["type"] == "known_item":
            tgt = set(q["target_urls"])
            r = next((h["rank"] for h in hits if h["url"] in tgt), None)
            ranks.append(r)
    if dump:
        dump.close()

    # ---- known item ----------------------------------------------------
    n = len(ranks)
    print(f"\nKNOWN ITEM  (n={n})")
    for k in (1, 3, 5, 10, 20):
        if k > args.k:
            continue
        hit = sum(1 for r in ranks if r is not None and r <= k)
        print(f"  Recall@{k:<3} {hit/n:.3f}   ({hit}/{n})")
    mrr = sum(1 / r for r in ranks if r is not None) / n
    print(f"  MRR@{args.k:<5} {mrr:.3f}")
    bad = [(q["qid"], r) for (q, _), r in zip(rows_by_type["known_item"], ranks)
           if r is None or r > 5]
    if bad:
        print(f"  outside top-5: " + ", ".join(
            f"{qid}@{'miss' if r is None else r}" for qid, r in bad))

    # ---- negatives vs known items --------------------------------------
    print("\nTOP-1 SCORE DISTRIBUTION")
    for t in ("known_item", "topical", "negative"):
        if t not in rows_by_type:
            continue
        s = np.array([hits[0]["score"] for _, hits in rows_by_type[t]])
        print(f"  {t:<11} n={len(s):<3} min {s.min():.3f}  "
              f"median {np.median(s):.3f}  max {s.max():.3f}")

    print("\nNEGATIVES (should have no good answer)")
    for q, hits in rows_by_type.get("negative", []):
        print(f"  {q['qid']} {hits[0]['score']:.3f}  {q['question']}")
        print(f"        -> {hits[0]['title'][:70]}")


if __name__ == "__main__":
    main()
