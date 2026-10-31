#!/usr/bin/env python3
"""Do near-identical articles get the same labels? An annotation-quality probe.

    .venv/bin/python dataset/label_consistency.py

Three DVB topic labels sit far below the rest -- social 0.27, ethnic 0.25,
religion 0.48 for the linear arm -- and fine-tuned XLM-R lifts them only to
0.31-0.37. The obvious temptation is to drop them, which raises macro-F1 by
about 0.07 and destroys the finding that explains the transformer's win, since
those same labels are where it gains most.

The alternative is to measure WHY they are low. The DVB labels are the
publisher's own editorial tags, so there is no second annotator and no
inter-annotator agreement to compute. But consistency can still be measured
without one: a news corpus covers each event many times, so if two
near-identical articles about the same event receive different labels, the
label is being applied inconsistently. That is measurable with no new
annotation.

METHOD. Articles are paired when they are published within WINDOW days of each
other and their TF-IDF cosine is at least THRESHOLD. Only the first HEAD
characters of each article are compared: a news event is identified by its
headline and lede, and later paragraphs diverge between outlets' follow-ups
even when the event is identical. Pairs are blocked by date because
same-event coverage is days apart, never years, and blocking is what makes
62k articles tractable.

For each label, over pairs where at least one article carries it:

    agreement = |both carry it| / |either carries it|

1.0 means the label is applied to near-duplicates consistently. 0.5 means that
half the time, one of two near-identical articles got the tag and the other did
not.

THE CONTROL THAT MAKES THIS WORK. Agreement rises with a label's prevalence
by chance alone -- for a label on a fraction p of articles, random assignment
gives p/(2-p). So the script reports that chance level per label, and the
comparison to lean on is between labels of SIMILAR prevalence and
DISSIMILAR F1. sport and social are the clean case: both sit near 4-5% of the
corpus, and their F1 differs by 0.7. If agreement tracks F1 there, prevalence
cannot be the explanation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from dataset import dvb_scheme as S  # noqa: E402
from dataset.tune_dvb import load  # noqa: E402

HEAD = 1200          # chars compared; a news event is its headline and lede
WINDOW = 3           # days; same-event coverage is days apart
CHUNK = 400


def parse_day(s):
    try:
        return date.fromisoformat((s or "")[:10]).toordinal()
    except Exception:
        return None


def find_pairs(M, days, order, thresholds):
    """Near-duplicate pairs, blocked by publication date.

    For each chunk of date-ordered articles, candidates are the chunk itself
    plus everything inside WINDOW days ahead. j > i keeps each pair once.
    """
    lo = min(thresholds)
    pairs, n = {t: [] for t in thresholds}, len(order)
    t0 = time.time()
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        end = b
        while end < n and days[order[end]] - days[order[b - 1]] <= WINDOW:
            end += 1
        sim = (M[order[a:b]] @ M[order[a:end]].T).toarray()
        for r in range(b - a):
            i_glob = a + r
            row = sim[r]
            hits = np.nonzero(row >= lo)[0]
            for c in hits:
                j_glob = a + c
                if j_glob <= i_glob:
                    continue
                if days[order[j_glob]] - days[order[i_glob]] > WINDOW:
                    continue
                for t in thresholds:
                    if row[c] >= t:
                        pairs[t].append((order[i_glob], order[j_glob]))
        if a and a % 8000 == 0:
            print(f"  {a}/{n}  {(time.time()-t0)/60:.1f}m  "
                  f"pairs@{lo} {len(pairs[lo])}", flush=True)
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset/myhumnews.jsonl"))
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.5, 0.6, 0.7])
    ap.add_argument("--out", type=Path,
                    default=Path("dataset/dvb_label_consistency.json"))
    args = ap.parse_args()

    rows = load(args.data)
    labels = sorted(S.F1_TEMPORAL, key=lambda l: -S.F1_TEMPORAL[l])
    text = [((d.get("title") or "") + "\n" + (d.get("body") or ""))[:HEAD]
            for d, _ in rows]
    Y = [set(ls) for _, ls in rows]
    days = [parse_day(d.get("published_date")) for d, _ in rows]

    keep = [i for i in range(len(rows)) if days[i] is not None]
    order = sorted(keep, key=lambda i: days[i])
    print(f"articles {len(rows)}   dated {len(order)}   "
          f"head {HEAD} chars   window +/-{WINDOW}d", flush=True)

    print("vectorising...", flush=True)
    t0 = time.time()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=120_000, min_df=5, sublinear_tf=True)
    M = vec.fit_transform(text).astype(np.float32)
    M = sparse.csr_matrix(M)
    print(f"  {M.shape} nnz {M.nnz/1e6:.0f}M  [{(time.time()-t0)/60:.1f}m]",
          flush=True)

    print("finding near-duplicate pairs...", flush=True)
    pairs = find_pairs(M, days, order, sorted(args.thresholds))

    prev = {l: sum(1 for s in Y if l in s) / len(Y) for l in labels}
    out = {"head_chars": HEAD, "window_days": WINDOW,
           "n_articles": len(rows), "n_dated": len(order),
           "f1_reference": dict(S.F1_TEMPORAL), "prevalence": prev,
           "by_threshold": {}}

    for t in sorted(args.thresholds):
        pr = pairs[t]
        print(f"\n=== cosine >= {t}:  {len(pr)} pairs ===")
        print(f"  {'label':<34}{'pairs':>7}{'agree':>8}{'chance':>8}"
              f"{'F1':>7}")
        rec = {}
        for l in labels:
            either = both = 0
            for i, j in pr:
                a, b = l in Y[i], l in Y[j]
                if a or b:
                    either += 1
                    both += a and b
            if either < 20:
                continue
            p = prev[l]
            rec[l] = {"pairs_either": either, "pairs_both": both,
                      "agreement": round(both / either, 4),
                      "chance": round(p / (2 - p), 4),
                      "f1_linear": S.F1_TEMPORAL[l]}
            print(f"  {l:<34}{either:>7}{both/either:>8.3f}"
                  f"{p/(2-p):>8.3f}{S.F1_TEMPORAL[l]:>7.3f}")
        out["by_threshold"][str(t)] = {"n_pairs": len(pr), "labels": rec}

        if len(rec) >= 4:
            ag = np.array([rec[l]["agreement"] for l in rec])
            f1 = np.array([rec[l]["f1_linear"] for l in rec])
            r = np.corrcoef(ag, f1)[0, 1]
            out["by_threshold"][str(t)]["corr_agreement_f1"] = round(float(r), 4)
            print(f"  corr(agreement, F1) = {r:+.3f}  over {len(rec)} labels")

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
