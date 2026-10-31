#!/usr/bin/env python3
"""Phase 5 -- label the whole corpus with the trained HumSet classifier.

Phase 4 measured the classifier and threw it away: train_humset.py cross
-validates, prints a number and keeps nothing. Nothing in this repo could
label an unseen article until this script. That is what it is for.

    .venv/bin/python -m classifier.predict_corpus

Model: TF-IDF char_wb(3,5) + one-vs-rest LinearSVC. Chosen over XLM-R on
the measurement, not on taste -- 0.755 vs 0.621 macro-F1 on full text, and
seconds rather than hours over 38k articles.

Training set: gold (383) + round 2's TRAINING half (180). Round 2's
held-out half is deliberately excluded so it stays usable as a test set.
The augmented arm measured better: macro over all 7 labels 0.539 -> 0.652.

THRESHOLDS. LinearSVC emits a signed margin, not a probability, and the
default cut at 0 is arbitrary for a rare label. Each label gets its own
threshold chosen to maximise F1 on OUT-OF-FOLD scores, never on scores the
model has already seen -- picking a cut on in-sample margins would be
fitting the training set twice.

TWO LABELS ARE NOT TRUSTWORTHY. `humanitarian_access` has 16 gold
positives and `information_communication` has 6. A threshold fitted to 6
examples is noise, and humanitarian_access additionally has no agreed
human boundary (kappa 0.091 between annotators). Both are still written
out, flagged `low_confidence`, and are EXCLUDED from `labels_filter` --
the field a retrieval filter should actually use. Nothing is discarded;
nothing unreliable silently drives retrieval.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier import humset                                    # noqa: E402
from classifier.train_humset import (MIN_SUPPORT, SEED, build,   # noqa: E402
                                     load_rows, prf, repair)

D = Path(__file__).resolve().parent / "data"
LOW_CONFIDENCE_MAX_SUPPORT = MIN_SUPPORT      # 20


def fit_all(X, Y, max_feat: int, C: float):
    """Fit the vectoriser and one binary SVC per label on everything."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=max_feat, sublinear_tf=True, min_df=2)
    A = vec.fit_transform(X)
    models = {}
    for lab in humset.LABELS:
        y = np.array([1 if lab in s else 0 for s in Y])
        if y.sum() == 0 or y.sum() == len(y):
            models[lab] = None                 # degenerate: no usable model
            continue
        clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000,
                        random_state=SEED)
        clf.fit(A, y)
        models[lab] = clf
    return vec, A, models


def oof_thresholds(X, Y, max_feat: int, C: float, folds: int):
    """Per-label threshold maximising F1 on out-of-fold margins."""
    sig = ["|".join(sorted(s)) for s in Y]
    counts = collections.Counter(sig)
    strat = [s if counts[s] >= folds else "_rare" for s in sig]
    oof = {lab: np.full(len(X), np.nan) for lab in humset.LABELS}

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, strat):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                              max_features=max_feat, sublinear_tf=True,
                              min_df=2)
        A = vec.fit_transform([X[i] for i in tr])
        B = vec.transform([X[i] for i in te])
        for lab in humset.LABELS:
            y = np.array([1 if lab in Y[i] else 0 for i in tr])
            if y.sum() == 0 or y.sum() == len(y):
                continue
            clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000,
                            random_state=SEED)
            clf.fit(A, y)
            for k, i in enumerate(te):
                oof[lab][i] = clf.decision_function(B[k])[0]

    thr, table = {}, []
    for lab in humset.LABELS:
        s = oof[lab]
        gold = np.array([1 if lab in Y[i] else 0 for i in range(len(X))])
        ok = ~np.isnan(s)
        sup = int(gold.sum())
        if sup == 0 or ok.sum() == 0:
            thr[lab] = 0.0
            table.append((lab, sup, 0.0, 0.0))
            continue
        best_t, best_f = 0.0, -1.0
        for t in np.unique(np.round(s[ok], 3)):
            p = (s >= t) & ok
            f = prf(int((p & (gold == 1)).sum()),
                    int((p & (gold == 0)).sum()),
                    int((~p & ok & (gold == 1)).sum()))[2]
            if f > best_f:
                best_f, best_t = f, float(t)
        thr[lab] = best_t
        table.append((lab, sup, best_t, best_f))
    return thr, table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=D / "humset_gold.jsonl")
    ap.add_argument("--task", type=Path, default=D / "humset_task.jsonl")
    ap.add_argument("--round2", type=Path, default=D / "humset_round2.jsonl")
    ap.add_argument("--round2-task", type=Path, default=D / "humset_thin_task.jsonl")
    ap.add_argument("--corpus", default="data-vm",
                    help="directory of <outlet>/parsed/articles.jsonl to label")
    # The full release (134,667 articles, 2021-2026) is one jsonl, not the
    # data-vm/<outlet>/parsed/ tree, so it needs its own reader.
    ap.add_argument("--corpus-jsonl", type=Path, default=None,
                    help="one jsonl of articles (e.g. dataset/myhumnews.jsonl); "
                         "overrides --corpus")
    ap.add_argument("--out", type=Path, default=D / "corpus_labels.jsonl")
    ap.add_argument("--max-features", type=int, default=300_000)
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-chars", type=int, default=200)
    args = ap.parse_args()

    X, Y, _ = build(args.gold, args.task)
    n_gold = len(X)
    print(f"gold            {len(X)}")
    if args.round2.exists() and args.round2_task.exists():
        Xa, Ya, _ = build(args.round2, args.round2_task, split="train")
        X, Y = X + Xa, Y + Ya
        print(f"+ round2 train  {len(Xa)}   (held-out half deliberately excluded)")
    print(f"training set    {len(X)}")

    # Trust is judged on GOLD support, never on the augmented training set.
    # Round 2 was deliberately oversampled for the thin labels, so counting it
    # here would hide exactly what this flag exists to catch: it lifts
    # humanitarian_access from 16 to 67 without making the label any more
    # agreed-upon (kappa 0.091 between human annotators).
    gold_support = collections.Counter(l for s in Y[:n_gold] for l in s)
    low = {l for l in humset.LABELS
           if gold_support[l] < LOW_CONFIDENCE_MAX_SUPPORT}

    print(f"\nfitting per-label thresholds on {args.folds}-fold out-of-fold margins")
    thr, table = oof_thresholds(X, Y, args.max_features, args.C, args.folds)
    print(f"\n  {'label':<28}{'gold':>6}{'train':>7}{'thresh':>9}"
          f"{'oofF1':>8}   trust")
    for lab, sup, t, f in table:
        print(f"  {lab:<28}{gold_support[lab]:>6}{sup:>7}{t:>9.3f}{f:>8.3f}   "
              f"{'LOW -- excluded from filter' if lab in low else 'ok'}")

    vec, _, models = fit_all(X, Y, args.max_features, args.C)

    rows = []
    files = ([args.corpus_jsonl] if args.corpus_jsonl
             else sorted(Path(args.corpus).glob("*/parsed/articles.jsonl")))
    for f in files:
        for line in open(f):
            d = json.loads(line)
            text = (d.get("title") or "") + "\n" + (d.get("body") or "")
            if len(text) < args.min_chars:
                continue
            rows.append((d["url"], d.get("outlet"),
                         (d.get("published_date") or "")[:10], text))
    print(f"\ncorpus          {len(rows)} articles with >= {args.min_chars} chars")

    B = vec.transform([t for _, _, _, t in rows])
    margins = {}
    for lab in humset.LABELS:
        m = models[lab]
        margins[lab] = (m.decision_function(B) if m is not None
                        else np.full(len(rows), -9.9))

    counts, counts_f = collections.Counter(), collections.Counter()
    n_filtered = 0
    with open(args.out, "w") as fh:
        for i, (url, outlet, date, _) in enumerate(rows):
            sc = {lab: float(margins[lab][i]) for lab in humset.LABELS}
            raw = {lab for lab in humset.LABELS if sc[lab] >= thr[lab]}
            labs = repair(sc, raw)
            filt = sorted(l for l in labs if l not in low)
            counts.update(labs)
            counts_f.update(filt)
            n_filtered += bool(filt)
            fh.write(json.dumps({
                "url": url, "outlet": outlet, "published_date": date,
                "labels": sorted(labs),
                "labels_filter": filt,
                "low_confidence": sorted(l for l in labs if l in low),
                "scores": {k: round(v, 4) for k, v in sc.items()},
            }, ensure_ascii=False) + "\n")

    print(f"\nwrote {args.out}  ({len(rows)} rows)")
    print(f"\n  {'label':<28}{'predicted':>10}{'% corpus':>10}   in filter")
    for lab in humset.LABELS:
        n = counts[lab]
        print(f"  {lab:<28}{n:>10}{100 * n / len(rows):>9.1f}%   "
              f"{'no (low confidence)' if lab in low else 'yes'}")
    # Calibration. The gold set was sampled from THIS corpus, so gold
    # prevalence is the population rate. A predicted rate far from it means
    # the label is being over- or under-called corpus-wide, which no F1 on a
    # held-out split would reveal.
    print(f"\n  {'label':<28}{'gold %':>8}{'pred %':>8}{'ratio':>8}")
    for lab in humset.LABELS:
        g = 100 * gold_support[lab] / n_gold
        q = 100 * counts[lab] / len(rows)
        flag = "" if g == 0 or 0.5 <= q / g <= 2.0 else "   <- off"
        print(f"  {lab:<28}{g:>7.1f}%{q:>7.1f}%{(q / g if g else 0):>8.2f}{flag}")

    print(f"\n  articles with >=1 filter label   {n_filtered}  "
          f"({100 * n_filtered / len(rows):.1f}%)")
    print(f"  articles labelled only {humset.EXCLUSIVE!r}  "
          f"{counts[humset.EXCLUSIVE]}  "
          f"({100 * counts[humset.EXCLUSIVE] / len(rows):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
