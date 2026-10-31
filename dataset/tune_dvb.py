#!/usr/bin/env python3
"""Hyperparameter sweep + per-label thresholds for the DVB classifier.

    .venv/bin/python -u -m dataset.tune_dvb

Two changes over dataset/train_dvb.py, both chosen because the current
settings were inherited from a classifier tuned on 743 humanitarian
articles and this task has 62,016:

  1. SWEEP C, max_features and min_df.
  2. PER-LABEL THRESHOLDS. LinearSVC fires at decision_function > 0, which
     is arbitrary for one-vs-rest on imbalanced classes -- a label with 200
     positives among 50,000 is pushed toward never firing. Each label gets
     the threshold maximising its F1.

Thresholds are fitted on a VALIDATION slice carved out of training data and
scored on a test set the threshold never saw. Fitting them on the test set
would be choosing the answer and then grading it.

Split is temporal throughout (train <=2024, test 2025+), the harsher and
more honest number for a transfer task.
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


def load(path: Path):
    rows = []
    for line in open(path):
        d = json.loads(line)
        if d["outlet"] != "dvb_burmese":
            continue
        labs = [l for l in S.labels(d.get("categories")) if l in S.F1_TEMPORAL]
        if labs:
            rows.append((d, labs))
    return rows


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def best_threshold(scores, gold):
    """Threshold maximising F1 on the validation slice."""
    best_t, best_f = 0.0, -1.0
    for t in np.unique(np.round(scores, 2)):
        pred = scores >= t
        f = prf(int((pred & (gold == 1)).sum()),
                int((pred & (gold == 0)).sum()),
                int((~pred & (gold == 1)).sum()))[2]
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def evaluate(Xtr, Ytr, Xva, Yva, Xte, Yte, labels, C, max_feat, min_df,
             tune_threshold, verbose=False):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=max_feat, sublinear_tf=True,
                          min_df=min_df)
    A = vec.fit_transform(Xtr)
    V = vec.transform(Xva)
    B = vec.transform(Xte)
    f1s, rows = [], []
    for lab in labels:
        ytr = np.array([1 if lab in s else 0 for s in Ytr])
        yva = np.array([1 if lab in s else 0 for s in Yva])
        yte = np.array([1 if lab in s else 0 for s in Yte])
        if ytr.sum() == 0 or yte.sum() == 0:
            continue
        clf = LinearSVC(C=C, class_weight="balanced", max_iter=3000,
                        random_state=SEED)
        clf.fit(A, ytr)
        thr = best_threshold(clf.decision_function(V), yva) if tune_threshold else 0.0
        pred = clf.decision_function(B) >= thr
        p, r, f = prf(int((pred & (yte == 1)).sum()),
                      int((pred & (yte == 0)).sum()),
                      int((~pred & (yte == 1)).sum()))
        f1s.append(f)
        rows.append((lab, int(yte.sum()), thr, p, r, f))
    macro = sum(f1s) / len(f1s)
    if verbose:
        print(f"\n  {'label':<34}{'n':>6}{'thr':>7}{'P':>7}{'R':>7}{'F1':>7}")
        for lab, n, t, p, r, f in rows:
            print(f"  {lab:<34}{n:>6}{t:>7.2f}{p:>7.3f}{r:>7.3f}{f:>7.3f}")
    return macro, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset/myhumnews.jsonl"))
    args = ap.parse_args()

    rows = load(args.data)
    labels = sorted(S.F1_TEMPORAL, key=lambda l: -S.F1_TEMPORAL[l])
    text = [(d.get("title") or "") + "\n" + (d.get("body") or "") for d, _ in rows]
    Y = [set(ls) for _, ls in rows]
    year = [(d.get("published_date") or "")[:4] for d, _ in rows]

    tr_all = [i for i in range(len(text)) if year[i] and year[i] <= "2024"]
    te = [i for i in range(len(text)) if year[i] and year[i] >= "2025"]
    tr, va = train_test_split(tr_all, test_size=0.15, random_state=SEED)
    print(f"articles {len(rows)}   train {len(tr)}  val {len(va)}  test {len(te)}")
    print(f"labels {len(labels)}   (temporal split, val carved from train)")

    X = lambda idx: [text[i] for i in idx]
    L = lambda idx: [Y[i] for i in idx]

    print("\n=== baseline (current settings, threshold at 0) ===")
    base, _ = evaluate(X(tr), L(tr), X(va), L(va), X(te), L(te),
                       labels, 0.5, 300_000, 3, False)
    print(f"  macro-F1 {base:.3f}")

    print("\n=== step 2: per-label thresholds, same settings ===")
    thr_only, _ = evaluate(X(tr), L(tr), X(va), L(va), X(te), L(te),
                           labels, 0.5, 300_000, 3, True)
    print(f"  macro-F1 {thr_only:.3f}   ({thr_only - base:+.3f})")

    print("\n=== step 3: sweep C / max_features / min_df (with thresholds) ===")
    print(f"  {'C':>5}{'max_feat':>10}{'min_df':>8}{'macro-F1':>10}")
    best = (thr_only, 0.5, 300_000, 3)
    for C in (0.25, 0.5, 1.0, 2.0):
        for mf in (300_000, 600_000):
            for md in (2, 3):
                if (C, mf, md) == (0.5, 300_000, 3):
                    m = thr_only
                else:
                    m, _ = evaluate(X(tr), L(tr), X(va), L(va), X(te), L(te),
                                    labels, C, mf, md, True)
                print(f"  {C:>5}{mf:>10}{md:>8}{m:>10.3f}")
                if m > best[0]:
                    best = (m, C, mf, md)

    m, C, mf, md = best
    print(f"\n=== best: C={C} max_features={mf} min_df={md} ===")
    final, frows = evaluate(X(tr), L(tr), X(va), L(va), X(te), L(te),
                            labels, C, mf, md, True, verbose=True)
    print(f"\n  macro-F1 {final:.3f}")
    print("\n" + "=" * 58)
    print(f"  baseline            {base:.3f}")
    print(f"  + thresholds        {thr_only:.3f}  ({thr_only - base:+.3f})")
    print(f"  + tuned params      {final:.3f}  ({final - base:+.3f} total)")
    print("=" * 58)
    json.dump({"C": C, "max_features": mf, "min_df": md,
               "macro_f1_temporal": round(final, 4),
               "thresholds": {r[0]: round(r[2], 3) for r in frows},
               "f1": {r[0]: round(r[5], 4) for r in frows}},
              open("dataset/dvb_model_config.json", "w"),
              ensure_ascii=False, indent=1)
    print("wrote dataset/dvb_model_config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
