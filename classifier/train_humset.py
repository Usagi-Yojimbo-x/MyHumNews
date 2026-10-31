#!/usr/bin/env python3
"""
TF-IDF one-vs-rest baseline for the 7 HumSet labels. MULTI-LABEL.

    python -m classifier.train_humset                      # 5-fold CV on the gold set
    python -m classifier.train_humset --train <llm.jsonl>  # train on pre-labels, test on gold

This is the TF-IDF arm of the proposal's TF-IDF-vs-XLM-R ablation.

FEATURES: char_wb 3-5 grams, as in the general-news baseline. Deliberately not
word n-grams: Burmese is written without spaces between words, so word
features would require a segmenter and inherit its errors. Character n-grams
sidestep segmentation entirely, which is also why "segment words" is a weak
ablation for this arm -- the features never see word boundaries either way.

WHY CROSS-VALIDATION AND NOT A TRAIN/TEST SPLIT
-----------------------------------------------
There are 383 hand-labelled articles and no LLM pre-labels yet. A 80/20 split
would train on 306 and test on 77, putting `information_communication` (6
positives in total) at an expected 1.2 positives in the test fold. The number
that came out would be noise.

5-fold CV uses every article for testing exactly once, so the reported scores
are over all 383. It does NOT fix the underlying problem: a label with 6
positives has ~4.8 in each training fold, and no model learns a class from
five examples. Those labels are reported as `thin` and are not results.

This is a SUPERVISION-STARVED number, not the proposal's intended TF-IDF arm.
The proposal trains on LLM pre-labels at corpus scale (38,152 articles) and
tests on these 383. Until those pre-labels exist this is a floor: what char
n-grams achieve from 383 examples.

GROUPING: folds are grouped by label-set signature so each fold sees a similar
mix of label combinations. sklearn has no multi-label stratifier; this is the
cheap approximation and it is honest about being one.

`other` IS EXCLUSIVE, and one-vs-rest does not know that. Seven independent
binary decisions can output `other` together with `casualties`, or output
nothing at all. Both are repaired after the fact, and the repair is reported
as an ablation because it is a real part of the system:

  raw        the seven decisions as they come out
  repaired   empty set -> highest-scoring single label
             `other` + anything -> drop whichever side scores lower overall
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from classifier import humset

MIN_SUPPORT = 20
SEED = 20261015


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def build(gold_path: Path, task_path: Path,
          split: str | None = None) -> tuple[list[str], list[set[str]], list[dict]]:
    """Join labels (gold) to text (task) on url.

    `split` keeps only rows whose "split" field matches -- used by --augment to
    take round 2's training half and leave its held-out half untouched.
    """
    text = {r["url"]: r for r in load_rows(task_path)}
    rows, X, Y = [], [], []
    for r in load_rows(gold_path):
        if split is not None and r.get("split") != split:
            continue
        t = text.get(r["url"])
        if t is None:
            sys.exit(f"no text for {r['url']} in {task_path}")
        labels = set(r["labels"])
        if not labels:
            continue
        X.append((t.get("title") or "") + "\n" + (t.get("body") or ""))
        Y.append(labels)
        rows.append({**r, "n_chars": t.get("n_chars"), "site": r.get("site")})
    return X, Y, rows


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def repair(scores: dict[str, float], raw: set[str]) -> set[str]:
    """Enforce `other` exclusivity and non-emptiness."""
    out = set(raw)
    if humset.EXCLUSIVE in out and len(out) > 1:
        others = out - {humset.EXCLUSIVE}
        # keep whichever side is more confident overall
        if scores[humset.EXCLUSIVE] >= max(scores[l] for l in others):
            out = {humset.EXCLUSIVE}
        else:
            out = others
    if not out:
        out = {max(scores, key=scores.get)}
    return out


def report(Y: list[set[str]], P: list[set[str]], rows: list[dict],
           title: str, show_floor: bool = True) -> float:
    n = len(Y)
    print(f"\n{'=' * 72}\n{title}  (n={n})\n{'=' * 72}")
    stats = {}
    for lab in humset.LABELS:
        tp = sum(1 for y, p in zip(Y, P) if lab in y and lab in p)
        fp = sum(1 for y, p in zip(Y, P) if lab not in y and lab in p)
        fn = sum(1 for y, p in zip(Y, P) if lab in y and lab not in p)
        stats[lab] = (tp, fp, fn, tp + fn)

    print(f"\n  {'label':28} {'P':>6} {'R':>6} {'F1':>6} {'supp':>6} {'pred':>6}")
    for lab in humset.LABELS:
        tp, fp, fn, sup = stats[lab]
        p, r, f = prf(tp, fp, fn)
        mark = "" if sup >= MIN_SUPPORT else "   thin"
        print(f"  {lab:28} {p:6.3f} {r:6.3f} {f:6.3f} {sup:6d} {tp + fp:6d}{mark}")

    f1s = {l: prf(*stats[l][:3])[2] for l in humset.LABELS}
    meas = [l for l in humset.LABELS if stats[l][3] >= MIN_SUPPORT]
    macro_all = sum(f1s.values()) / len(f1s)
    macro_meas = sum(f1s[l] for l in meas) / len(meas)
    TP = sum(s[0] for s in stats.values())
    FP = sum(s[1] for s in stats.values())
    FN = sum(s[2] for s in stats.values())
    micro = prf(TP, FP, FN)[2]
    exact = sum(1 for y, p in zip(Y, P) if y == p)
    lo, hi = wilson(exact, n)
    jac = sum(len(y & p) / len(y | p) for y, p in zip(Y, P) if y | p) / n
    ham = sum(len(y ^ p) for y, p in zip(Y, P)) / (n * len(humset.LABELS))

    print(f"\n  macro-F1 (all 7)              {macro_all:.3f}")
    print(f"  macro-F1 (support >= {MIN_SUPPORT}, n={len(meas)})  {macro_meas:.3f}   <-- headline")
    print(f"  micro-F1                      {micro:.3f}")
    print(f"  exact match                   {exact / n:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  Jaccard                       {jac:.3f}")
    print(f"  Hamming loss                  {ham:.4f}")

    if show_floor:
        freq = collections.Counter(l for y in Y for l in y)
        top2 = {l for l, _ in freq.most_common(2)}
        bf = {l: prf(sum(1 for y in Y if l in y and l in top2),
                     sum(1 for y in Y if l not in y and l in top2),
                     sum(1 for y in Y if l in y and l not in top2))[2]
              for l in humset.LABELS}
        bmeas = sum(bf[l] for l in meas) / len(meas)
        print(f"\n  floor, always predict {sorted(top2)}:")
        print(f"    macro-F1 (all 7) {sum(bf.values()) / len(bf):.3f}   "
              f"(support >= {MIN_SUPPORT}) {bmeas:.3f}   "
              f"exact {sum(1 for y in Y if y == top2) / n:.3f}")
        print(f"  --> beats floor by {macro_meas - bmeas:+.3f} macro-F1 on measurable labels")

    print("\n  per outlet (macro-F1 over measurable labels):")
    for site in sorted({humset.canon_site(r["site"]) for r in rows}):
        idx = [i for i, r in enumerate(rows)
               if humset.canon_site(r["site"]) == site]
        sf = []
        for lab in meas:
            tp = sum(1 for i in idx if lab in Y[i] and lab in P[i])
            fp = sum(1 for i in idx if lab not in Y[i] and lab in P[i])
            fn = sum(1 for i in idx if lab in Y[i] and lab not in P[i])
            sf.append(prf(tp, fp, fn)[2])
        ex = sum(1 for i in idx if Y[i] == P[i]) / len(idx)
        print(f"    {site:5} n={len(idx):4d}  macro-F1 {sum(sf) / len(sf):.3f}  exact {ex:.3f}")

    conf = collections.Counter()
    for y, p in zip(Y, P):
        for g in y - p:
            for q in p - y:
                conf[(g, q)] += 1
    if conf:
        print("\n  confusions (gold -> predicted instead):")
        for (g, q), c in conf.most_common(8):
            print(f"    {c:4d} ({c / stats[g][3]:5.1%} of {g:26}) {g} -> {q}")
    return macro_meas


def fit_predict(Xtr, Ytr, Xte, max_feat: int, C: float):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=max_feat, sublinear_tf=True, min_df=2)
    A = vec.fit_transform(Xtr)
    B = vec.transform(Xte)
    raw, scores = [set() for _ in Xte], [dict() for _ in Xte]
    for lab in humset.LABELS:
        y = np.array([1 if lab in s else 0 for s in Ytr])
        if y.sum() == 0:                      # label absent from this fold
            for d in scores:
                d[lab] = -9.9
            continue
        if y.sum() == len(y):
            for i, d in enumerate(scores):
                d[lab] = 9.9
                raw[i].add(lab)
            continue
        clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000,
                        random_state=SEED)
        clf.fit(A, y)
        dec = clf.decision_function(B)
        for i, s in enumerate(dec):
            scores[i][lab] = float(s)
            if s > 0:
                raw[i].add(lab)
    return raw, scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=Path("classifier/data/humset_gold.jsonl"))
    ap.add_argument("--task", type=Path, default=Path("classifier/data/humset_task.jsonl"))
    ap.add_argument("--train", type=Path,
                    help="separate training file (LLM pre-labels). "
                         "Without it, 5-fold CV on --gold.")
    ap.add_argument("--train-task", type=Path,
                    help="text for --train, if not in the same file")
    ap.add_argument("--folds", type=int, default=5)
    # Non-binding at gold-set size: char_wb 3-5 with min_df=2 over 383
    # articles builds only 65,955 n-grams. Kept at 300k for the corpus-scale
    # run, where it does bind.
    ap.add_argument("--max-features", type=int, default=300_000)
    # C=0.5, not 1.0. Chosen by a PAIRED comparison across 5 different CV
    # splits, where C=0.5 won 5/5 (mean +0.015 macro-F1, sd 0.004). Within a
    # single split the fold-to-fold sd is 0.055, so a one-off sweep could not
    # have justified this -- the paired-across-splits test could.
    ap.add_argument("-C", type=float, default=0.5)
    ap.add_argument("--augment", type=Path,
                    help="extra LABELLED articles appended to every CV fold's "
                         "TRAINING split only. The test folds stay pure gold, so "
                         "the score still describes the population gold was "
                         "sampled from. Ignored when --train is given.")
    ap.add_argument("--augment-task", type=Path,
                    help="text source for --augment (it carries no bodies)")
    ap.add_argument("--augment-split", default="train",
                    help="keep only --augment rows with this `split` value; "
                         "pass an empty string to use all of them")
    ap.add_argument("--dump", type=Path, help="write predictions as jsonl")
    args = ap.parse_args()

    X, Y, rows = build(args.gold, args.task)
    print(f"gold {len(X)} articles · "
          f"{sum(len(y) for y in Y) / len(Y):.2f} labels/article · "
          f"char_wb 3-5, max_features={args.max_features:,}, C={args.C}")

    Xa: list[str] = []
    Ya: list[set[str]] = []
    if args.augment:
        if args.train:
            sys.exit("--augment and --train are mutually exclusive: --train "
                     "replaces cross-validation, --augment adds to its folds")
        Xa, Ya, arows = build(args.augment, args.augment_task or args.augment,
                              split=args.augment_split or None)
        overlap = {r["url"] for r in arows} & {r["url"] for r in rows}
        if overlap:
            sys.exit(f"--augment leaks into the test set: {len(overlap)} shared "
                     f"urls, e.g. {sorted(overlap)[:3]}")
        apos = {l: sum(1 for y in Ya if l in y) for l in humset.LABELS}
        print(f"augment +{len(Xa)} articles into every training fold "
              f"(split={args.augment_split or 'all'}, 0 overlap with gold)")
        for l in humset.LABELS:
            g = sum(1 for y in Y if l in y)
            print(f"    {l:<28} gold {g:>4}  +augment {apos[l]:>4}"
                  f"  -> train pool {int(g * (args.folds - 1) / args.folds) + apos[l]:>4}")

    if args.train:
        Xtr, Ytr, _ = build(args.train, args.train_task or args.train)
        print(f"training on {len(Xtr)} pre-labelled articles")
        raw, scores = fit_predict(Xtr, Ytr, X, args.max_features, args.C)
    else:
        print(f"{args.folds}-fold CV, grouped by label-set signature "
              f"(no separate training pool exists yet)")
        sig = ["|".join(sorted(y)) for y in Y]
        # collapse signatures too rare to stratify on
        cnt = collections.Counter(sig)
        sig = [s if cnt[s] >= args.folds else "RARE" for s in sig]
        print(f"  {len(set(sig))} signature groups "
              f"({sum(1 for s in sig if s == 'RARE')} articles in RARE)")
        raw = [None] * len(X)
        scores = [None] * len(X)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
        for k, (tr, te) in enumerate(skf.split(X, sig), 1):
            r, s = fit_predict([X[i] for i in tr] + Xa,
                               [Y[i] for i in tr] + Ya,
                               [X[i] for i in te], args.max_features, args.C)
            for j, i in enumerate(te):
                raw[i], scores[i] = r[j], s[j]
            print(f"  fold {k}: train {len(tr) + len(Xa)} test {len(te)}")

    rep = [repair(s, r) for s, r in zip(scores, raw)]

    n_empty = sum(1 for r in raw if not r)
    n_viol = sum(1 for r in raw if humset.EXCLUSIVE in r and len(r) > 1)
    m_raw = report(Y, raw, rows, "RAW — seven independent binary decisions")
    print(f"\n  invalid outputs one-vs-rest produced: "
          f"{n_empty} empty ({n_empty / len(raw):.1%}), "
          f"{n_viol} '{humset.EXCLUSIVE}'+other ({n_viol / len(raw):.1%})")
    m_rep = report(Y, rep, rows,
                   "REPAIRED — empty set filled, `other` exclusivity enforced",
                   show_floor=True)
    print(f"\n{'=' * 72}\n  repair is worth {m_rep - m_raw:+.3f} macro-F1 "
          f"on measurable labels\n{'=' * 72}")

    if args.dump:
        with open(args.dump, "w") as fh:
            for r, y, p, s in zip(rows, Y, rep, scores):
                fh.write(json.dumps({
                    "url": r["url"], "site": r["site"],
                    "labels": sorted(y), "pred": sorted(p),
                    "scores": {k: round(v, 4) for k, v in s.items()},
                }, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
