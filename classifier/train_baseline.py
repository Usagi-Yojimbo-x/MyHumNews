#!/usr/bin/env python3
"""
Baseline topic classifier for the Burmese news corpus.

    python -m classifier.train_baseline

Character n-grams + TF-IDF + linear SVM. Chosen because Burmese has no spaces
between words, so every whitespace-tokenising pipeline fails on it. Character
n-grams sidestep word segmentation entirely and are a strong baseline for
Burmese classification -- establish this number before reaching for a
segmenter or a transformer.

What this script reports, and why each number is here:

  overall accuracy / macro-F1   headline numbers. Macro-F1 matters more than
                                accuracy: the classes run 34:1 imbalanced, so
                                accuracy is flattered by the big classes.

  per-class                     where it actually works. A class with high
                                support and low recall is a real failure; a
                                class with 73 test examples is noise.

  PER-OUTLET accuracy           the honest test. Training is ~80% DVB and the
                                classes ARE DVB's own section names, so a model
                                can score well by learning DVB's house style
                                instead of topic. If BBC and Irrawaddy accuracy
                                collapse relative to DVB, that is what happened.

  majority / stratified baselines  a classifier must beat these to mean
                                anything. Quoting accuracy without them is how
                                a 40%-accurate model gets reported as a result.

  confusion pairs               which topics bleed into each other. For this
                                taxonomy expect conflict<->politics and
                                international<->conflict, since Myanmar's civil
                                war is simultaneously all three.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from classifier.dataset_guard import check_dataset

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path: str):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(HERE, "data"))
    p.add_argument("--ngram-min", type=int, default=3)
    p.add_argument("--ngram-max", type=int, default=5)
    p.add_argument("--max-features", type=int, default=300_000)
    p.add_argument("--max-chars", type=int, default=3000,
                   help="truncate each article; news puts the topic up front")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--dump-pred", metavar="PATH", default=None,
                   help="write held-out predictions (url, outlet_label, pred, margin) "
                        "so hand labels can be scored against them")
    a = p.parse_args()

    check_dataset(a.data)
    train = load(os.path.join(a.data, "train.jsonl"))
    test = load(os.path.join(a.data, "test.jsonl"))
    if not train or not test:
        print("no data -- run classifier.build_dataset first", file=sys.stderr)
        return 1

    # Title is prepended: it is the densest topical signal in a news article.
    def feats(r):
        return (r["title"] + "\n" + r["text"])[:a.max_chars]

    Xtr = [feats(r) for r in train]
    ytr = [r["topic"] for r in train]
    Xte = [feats(r) for r in test]
    yte = [r["topic"] for r in test]

    print(f"train {len(Xtr)}  test {len(Xte)}  classes {len(set(ytr))}")
    print(f"features: char_wb {a.ngram_min}-{a.ngram_max} grams, "
          f"max {a.max_features:,}, first {a.max_chars} chars\n")

    # --- baselines to beat ---
    for strat in ("most_frequent", "stratified"):
        d = DummyClassifier(strategy=strat, random_state=0).fit(Xtr, ytr)
        pr = d.predict(Xte)
        print(f"  baseline {strat:14} acc={accuracy_score(yte, pr):.3f} "
              f"macroF1={f1_score(yte, pr, average='macro', zero_division=0):.3f}")

    # char_wb respects token boundaries where they exist (Latin/numerals) while
    # still sliding across Burmese, which has none.
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb",
                        ngram_range=(a.ngram_min, a.ngram_max),
                        max_features=a.max_features,
                        min_df=3, sublinear_tf=True),
        LinearSVC(C=a.C, class_weight="balanced", dual="auto", max_iter=5000),
    )

    t0 = time.time()
    pipe.fit(Xtr, ytr)
    fit_s = time.time() - t0
    pred = pipe.predict(Xte)

    acc = accuracy_score(yte, pred)
    mf1 = f1_score(yte, pred, average="macro", zero_division=0)
    print(f"\n=== model (fit {fit_s:.0f}s) ===")
    print(f"  accuracy  {acc:.3f}")
    print(f"  macro-F1  {mf1:.3f}")
    vocab = len(pipe.named_steps["tfidfvectorizer"].vocabulary_)
    print(f"  vocabulary {vocab:,} n-grams")

    print("\n=== per class ===")
    print(classification_report(yte, pred, zero_division=0, digits=3))

    # --- the honest test: does it generalise past DVB? ---
    print("=== per outlet (test) ===")
    print(f"  {'outlet':12} {'n':>6} {'acc':>7} {'macroF1':>8}")
    by = collections.defaultdict(list)
    for r, p_, t_ in zip(test, pred, yte):
        by[r["site"]].append((p_, t_))
    for site, pairs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        pp = [x[0] for x in pairs]
        tt = [x[1] for x in pairs]
        print(f"  {site:12} {len(pairs):>6} {accuracy_score(tt, pp):>7.3f} "
              f"{f1_score(tt, pp, average='macro', zero_division=0):>8.3f}")

    if a.dump_pred:
        # Margin = gap between the best and second-best class score, the same
        # confidence proxy predict_unlabelled uses. LinearSVC has no
        # probabilities.
        import numpy as np
        sc = pipe.decision_function(Xte)
        order = np.argsort(sc, axis=1)
        idx = np.arange(sc.shape[0])
        gap = sc[idx, order[:, -1]] - sc[idx, order[:, -2]]
        with open(a.dump_pred, "w", encoding="utf-8") as fh:
            for r, p_, g in zip(test, pred, gap):
                fh.write(json.dumps({
                    "url": r["url"], "site": r["site"],
                    "published_date": r.get("published_date"),
                    "outlet_label": r["topic"], "topic": p_,
                    "margin": round(float(g), 4),
                }, ensure_ascii=False) + "\n")
        print(f"\nwrote {a.dump_pred}  ({len(test)} held-out predictions)")

    print("\n=== biggest confusions (true -> pred) ===")
    labels = sorted(set(ytr) | set(yte))
    cm = confusion_matrix(yte, pred, labels=labels)
    off = []
    for i, t_ in enumerate(labels):
        for j, p_ in enumerate(labels):
            if i != j and cm[i][j]:
                off.append((cm[i][j], t_, p_))
    for n, t_, p_ in sorted(off, reverse=True)[:10]:
        tot = sum(cm[labels.index(t_)])
        print(f"  {n:>4} ({n/tot*100:4.1f}% of {t_})  {t_} -> {p_}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
