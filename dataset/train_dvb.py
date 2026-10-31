#!/usr/bin/env python3
"""Train a topic classifier on DVB's own editorial categories.

    .venv/bin/python -m dataset.train_dvb

Labels come from dataset/dvb_scheme.py -- DVB's real desks, not an invented
taxonomy. Multi-label: 12% of articles carry more than one desk.

Model: TF-IDF char_wb(3,5) + one-vs-rest LinearSVC, the same setup that beat
XLM-R on this corpus for the humanitarian labels (0.755 vs 0.621). Character
n-grams matter because Burmese has no reliable word segmentation.

TWO SPLITS, because one of them flatters the model:

  random    80/20 stratified on the label-set signature. The standard number.
  temporal  train on <=2024, test on 2025+. Harsher and more honest, because
            several DVB desks are time-bound -- sport has 1 article before
            2023, COVID19 runs 1,100 in 2021 down to 1 in 2026. On a random
            split the model can lean on date-correlated vocabulary; on a
            temporal split it cannot.

The gap between the two is the part worth reporting: it measures how much of
the score is really topic recognition rather than period recognition.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import dvb_scheme as S   # noqa: E402

SEED = 20261020


def load(path: Path, min_support: int, only=None):
    rows = []
    for line in open(path):
        d = json.loads(line)
        if d["outlet"] != "dvb_burmese":
            continue
        labs = S.labels(d.get("categories"))
        if only is not None:
            labs = [l for l in labs if l in only]
        if labs:
            rows.append((d, labs))
    support = collections.Counter(l for _, ls in rows for l in ls)
    keep = {l for l, v in support.items() if v >= min_support}
    out = []
    for d, ls in rows:
        ls = [l for l in ls if l in keep]
        if ls:
            out.append((d, ls))
    return out, sorted(keep, key=lambda l: -support[l])


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def run(Xtr, Ytr, Xte, Yte, labels, max_feat, C, title):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=max_feat, sublinear_tf=True, min_df=3)
    A = vec.fit_transform(Xtr)
    B = vec.transform(Xte)
    print(f"\n{title}")
    print(f"  train {len(Xtr)}  test {len(Xte)}  features {A.shape[1]}")
    print(f"\n  {'label':<34}{'support':>8}{'P':>7}{'R':>7}{'F1':>7}")
    f1s, rows = [], []
    for lab in labels:
        ytr = np.array([1 if lab in s else 0 for s in Ytr])
        yte = np.array([1 if lab in s else 0 for s in Yte])
        if ytr.sum() == 0 or yte.sum() == 0:
            continue
        clf = LinearSVC(C=C, class_weight="balanced", max_iter=3000,
                        random_state=SEED)
        clf.fit(A, ytr)
        pred = (clf.decision_function(B) > 0).astype(int)
        tp = int(((pred == 1) & (yte == 1)).sum())
        fp = int(((pred == 1) & (yte == 0)).sum())
        fn = int(((pred == 0) & (yte == 1)).sum())
        p, r, f = prf(tp, fp, fn)
        f1s.append(f)
        rows.append((lab, int(yte.sum()), p, r, f))
        print(f"  {lab:<34}{int(yte.sum()):>8}{p:>7.3f}{r:>7.3f}{f:>7.3f}")
    macro = sum(f1s) / len(f1s)
    print(f"\n  macro-F1 over {len(f1s)} labels   {macro:.3f}")
    return macro, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset/myhumnews.jsonl"))
    ap.add_argument("--min-support", type=int, default=S.MIN_SUPPORT)
    ap.add_argument("--max-features", type=int, default=300_000)
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--usable", action="store_true",
                    help="restrict to dvb_scheme.HIGH_CONFIDENCE (8 labels, F1>=0.70)")
    args = ap.parse_args()

    only = set(S.HIGH_CONFIDENCE) if args.usable else None
    rows, labels = load(args.data, args.min_support, only)
    print(f"dvb labelled articles {len(rows)}   labels {len(labels)}")
    n_multi = sum(1 for _, ls in rows if len(ls) > 1)
    print(f"multi-label articles  {n_multi} ({100*n_multi/len(rows):.0f}%)")

    text = [(d.get("title") or "") + "\n" + (d.get("body") or "") for d, _ in rows]
    Y = [set(ls) for _, ls in rows]
    year = [(d.get("published_date") or "")[:4] for d, _ in rows]

    # --- random split -------------------------------------------------
    sig = ["|".join(sorted(s)) for s in Y]
    cnt = collections.Counter(sig)
    strat = [s if cnt[s] >= 2 else "_rare" for s in sig]
    itr, ite = train_test_split(range(len(text)), test_size=0.2,
                                random_state=SEED, stratify=strat)
    m_rand, _ = run([text[i] for i in itr], [Y[i] for i in itr],
                    [text[i] for i in ite], [Y[i] for i in ite],
                    labels, args.max_features, args.C,
                    "=== RANDOM SPLIT (80/20, stratified) ===")

    # --- temporal split -----------------------------------------------
    jtr = [i for i in range(len(text)) if year[i] and year[i] <= "2024"]
    jte = [i for i in range(len(text)) if year[i] and year[i] >= "2025"]
    m_time, _ = run([text[i] for i in jtr], [Y[i] for i in jtr],
                    [text[i] for i in jte], [Y[i] for i in jte],
                    labels, args.max_features, args.C,
                    "=== TEMPORAL SPLIT (train <=2024, test 2025+) ===")

    print("\n" + "=" * 62)
    print(f"  random   macro-F1 {m_rand:.3f}")
    print(f"  temporal macro-F1 {m_time:.3f}")
    print(f"  gap               {m_rand - m_time:+.3f}"
          "   <- how much of the score is period, not topic")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
