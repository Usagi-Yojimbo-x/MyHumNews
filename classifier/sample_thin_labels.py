#!/usr/bin/env python3
"""
Build a labelling task targeted at the labels that have too few examples.

    python -m classifier.sample_thin_labels --n 180

WHY TARGETED AND NOT RANDOM
---------------------------
`humanitarian_access` has 16 positives in the gold set and
`information_communication` has 6. Both score F1 0.000. The learning curve
(README Phase 7) shows this is scarcity, not difficulty: every label in the
set, including the easiest one, scores ~0.00 at 5 positives and under 0.25 at
15. Reaching F1 ~0.5 needs roughly 30 positives.

At their base rates of 4.2% and 1.6%, uniform random labelling would need
~700 and ~1,900 more articles to find 30 positives each. Nobody is hand-reading
2,600 articles. Ranking the corpus first is what makes this affordable.

HOW IT RANKS, AND THE ASSUMPTION
--------------------------------
For each thin label, the one-vs-rest scorer is trained on all 383 gold articles
for that label alone and used to rank the whole corpus by decision margin; the
top `--n` are taken. This is most-confident-positive sampling.

The assumption is explicit and unverified: **these two classifiers score F1
0.000, so their ranking has never been shown to work.** The reason to expect it
might anyway is the Phase 4 finding that margin ORDERING transferred across
outlets even where calibration did not. The reason to doubt it is that a
classifier with zero recall has no positives to have learned from.

Either way the task reports the realised hit rate once labelled, so one pass
settles it. If the hit rate is near the 4.2%/1.6% base rate, ranking added
nothing and the fallback is keyword seeding from the existing positives.

WHAT THE ANNOTATOR MUST BE TOLD
-------------------------------
This sample is deliberately NOT representative. It is enriched for two rare
labels, so the label distribution will look nothing like the first 383. The
instruction is unchanged -- label what is actually there, including `other` --
because a biased sample labelled honestly is still usable for training, whereas
a sample labelled to match an expectation is not.

These articles must go into a TRAINING pool, not into `humset_gold.jsonl`.
Adding an enriched sample to the test set would silently change every base rate
the Phase 7 numbers rest on.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from classifier import humset, train_humset as T
from classifier.weak_supervision import load_corpus

GOLD = Path("classifier/data/humset_gold.jsonl")
TASK = Path("classifier/data/humset_task.jsonl")
THIN = ["humanitarian_access", "information_communication"]
R2 = Path("classifier/data/humset_round2.jsonl")
R2_TASK = Path("classifier/data/humset_thin_task.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    """Same row shape as weak_supervision.load_corpus, from one jsonl."""
    rows, seen = [], set()
    for line in open(path):
        r = json.loads(line)
        u, b = r.get("url"), (r.get("body") or "")
        if not u or u in seen or len(b) < 200:
            continue
        seen.add(u)
        rows.append({"url": u, "site": r.get("outlet"),
                     "title": r.get("title") or "", "body": b,
                     "published_date": r.get("published_date")})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=180,
                    help="articles per thin label")
    ap.add_argument("--out", type=Path,
                    default=Path("classifier/data/humset_thin_task.jsonl"))
    # Round 3 (2026-10-30): rank the full 2021-2026 release, not data-vm/.
    # Gold has no article before 2025, so drawing round 3 from 2021-2024 adds
    # old-period coverage and rare-label positives in the same labelling pass.
    ap.add_argument("--corpus-jsonl", type=Path, default=None,
                    help="rank this jsonl (e.g. dataset/myhumnews.jsonl) "
                         "instead of data-vm/")
    ap.add_argument("--years", default=None,
                    help="keep only these publication years, e.g. 2021-2024")
    ap.add_argument("--with-round2", action="store_true",
                    help="train the ranker on gold + round 2's training half, "
                         "and exclude all round-2 articles from the candidates")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="if set, mark half of each target label's picks "
                         "'train' and half 'test', at random")
    args = ap.parse_args()

    X, Y, _ = T.build(GOLD, TASK)
    labelled = {json.loads(l)["url"] for l in open(GOLD)}
    if args.with_round2:
        Xa, Ya, _ = T.build(R2, R2_TASK, split="train")
        X, Y = X + Xa, Y + Ya
        labelled |= {json.loads(l)["url"] for l in open(R2)}
        print(f"ranker trained on gold + round-2 train: {len(X)}")
    print("loading corpus ...", flush=True)
    corpus = load_jsonl(args.corpus_jsonl) if args.corpus_jsonl else load_corpus()
    if args.years:
        lo, hi = args.years.split("-")
        corpus = [r for r in corpus
                  if lo <= (r["published_date"] or "")[:4] <= hi]
    cand = [r for r in corpus if r["url"] not in labelled]
    print(f"{len(cand):,} candidate articles (already-labelled excluded)")

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=300_000, sublinear_tf=True, min_df=2)
    A = vec.fit_transform(X)
    B = vec.transform([r["title"] + "\n" + r["body"] for r in cand])

    picked: dict[str, str] = {}      # url -> which label pulled it in
    ranks: dict[str, float] = {}
    for label in THIN:
        pos = sum(1 for y in Y if label in y)
        clf = LinearSVC(C=0.5, class_weight="balanced", max_iter=5000,
                        random_state=T.SEED)
        clf.fit(A, np.array([1 if label in y else 0 for y in Y]))
        s = clf.decision_function(B)
        order = s.argsort()[::-1]
        print(f"\n  {label}: trained on {pos} positives "
              f"(base rate {pos / len(X):.1%})")
        print(f"    corpus scored > 0: {(s > 0).sum():,} of {len(cand):,} "
              f"({(s > 0).mean():.2%})")
        taken = 0
        for i in order:
            if taken >= args.n:
                break
            u = cand[i]["url"]
            if u in picked:
                continue
            picked[u] = label
            ranks[u] = float(s[i])
            taken += 1
        print(f"    took {taken} (margin {ranks[cand[order[0]]['url']]:.2f} .. "
              f"{min(ranks[u] for u in picked if picked[u] == label):.2f})")

    by_url = {r["url"]: r for r in cand}
    split = {}
    if args.split_seed is not None:
        rng = np.random.default_rng(args.split_seed)
        for label in THIN:
            us = sorted(u for u in picked if picked[u] == label)
            rng.shuffle(us)
            for k, u in enumerate(us):
                split[u] = "train" if k < len(us) // 2 else "test"
    rows = []
    for j, (u, label) in enumerate(sorted(picked.items(), key=lambda kv: -ranks[kv[0]]), 1):
        r = by_url[u]
        rows.append({"i": j, "url": u, "site": r["site"], "title": r["title"],
                     "body": r["body"], "published_date": r["published_date"],
                     "n_chars": len(r["body"]),
                     "_targeted_for": label, "_margin": round(ranks[u], 4),
                     **({"_split": split[u]} if split else {})})
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {args.out}: {len(rows)} articles")
    print(f"  by target label: {dict(collections.Counter(r['_targeted_for'] for r in rows))}")
    print(f"  by outlet:       {dict(collections.Counter(r['site'] for r in rows))}")
    print(f"  by year:         {dict(sorted(collections.Counter((r['published_date'] or '')[:4] for r in rows).items()))}")
    if split:
        print(f"  split:           {dict(collections.Counter((r['_targeted_for'], r['_split']) for r in rows))}")
    print(f"  mean length:     {sum(r['n_chars'] for r in rows) / len(rows):,.0f} chars")
    print("\n  NOTE: enriched sample, NOT representative. Goes into a TRAINING")
    print("  pool, never into humset_gold.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
