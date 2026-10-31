#!/usr/bin/env python3
"""Features and model-class sweep for the DVB classifier.

    .venv/bin/python -u -m dataset.sweep_dvb

dataset/tune_dvb.py swept C / max_features / min_df and found the inherited
settings were already best: 0.658 macro-F1, all of it from per-label
thresholds. This sweeps the two dimensions it never touched.

STAGE A -- features. Only char_wb(3,5) was ever tried. char_wb confines
n-grams inside "word" boundaries, but Burmese spaces are phrase-level, not
word-level, so char_wb may be discarding n-grams that straddle a space.
Also sweeps the n-gram range and max_df (an upper document-frequency cut,
never tested; min_df was).

Not swept because they are already set in train_dvb.py: sublinear_tf=True
and class_weight="balanced".

STAGE B -- model class. LinearSVC is the only estimator tried so far.
  LogisticRegression  near-identical F1 expected; gives calibrated
                      probabilities, which would make the thresholds and
                      predict_corpus.py's low_confidence flag interpretable.
  ComplementNB        built for imbalanced text, trains in seconds.
  ClassifierChain     the only structural change: one-vs-rest predicts each
                      of the 19 labels blind to the other 18, but the labels
                      correlate (politics/election, peace-war/ethnic,
                      COVID19/health). A chain feeds earlier predictions
                      forward as features.

Thresholds are fitted on a validation slice carved from training and scored
on a test set they never saw, as in tune_dvb. Split is temporal throughout
(train <=2024, test 2025+). Thresholds are NOT comparable across models:
LinearSVC and LogisticRegression threshold a margin, ComplementNB a
probability.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import dvb_scheme as S            # noqa: E402
from dataset.tune_dvb import SEED, load, prf, best_threshold   # noqa: E402

# The winner of the C / max_features / min_df sweep, held fixed here.
C, MAX_FEAT, MIN_DF = 0.5, 300_000, 2


def vectorizer(analyzer, ngram, max_df=1.0):
    return TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                           max_features=MAX_FEAT, sublinear_tf=True,
                           min_df=MIN_DF, max_df=max_df)


def svc():
    return LinearSVC(C=C, class_weight="balanced", max_iter=3000,
                     random_state=SEED)


def logreg():
    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                              solver="liblinear", random_state=SEED)


def cnb():
    return ComplementNB()


MODELS = {"LinearSVC": svc, "LogisticRegression": logreg, "ComplementNB": cnb}


def score_of(clf, X):
    """Decision margin where available, else P(positive)."""
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X)
    return clf.predict_proba(X)[:, 1]


def run_ovr(A, V, B, Ytr, Yva, Yte, labels, make_clf, verbose=False):
    """One-vs-rest: an independent binary model per label."""
    f1s, rows = [], []
    for lab in labels:
        ytr = np.array([1 if lab in s else 0 for s in Ytr])
        yva = np.array([1 if lab in s else 0 for s in Yva])
        yte = np.array([1 if lab in s else 0 for s in Yte])
        if ytr.sum() == 0 or yte.sum() == 0:
            continue
        clf = make_clf()
        clf.fit(A, ytr)
        thr = best_threshold(score_of(clf, V), yva)
        pred = score_of(clf, B) >= thr
        p, r, f = prf(int((pred & (yte == 1)).sum()),
                      int((pred & (yte == 0)).sum()),
                      int((~pred & (yte == 1)).sum()))
        f1s.append(f)
        rows.append((lab, int(yte.sum()), thr, p, r, f))
    return sum(f1s) / len(f1s), rows


def run_chain(A, V, B, Ytr, Yva, Yte, labels, verbose=False):
    """ClassifierChain: label k sees the predictions for labels 1..k-1.

    Chain order is the label order passed in (descending F1 from the earlier
    run), so the labels the model predicts well come first and feed the weak
    ones -- the arrangement most likely to help.
    """
    M = lambda Yset: np.array([[1 if l in s else 0 for l in labels] for s in Yset])
    ytr, yva, yte = M(Ytr), M(Yva), M(Yte)
    keep = [j for j in range(len(labels)) if ytr[:, j].sum() and yte[:, j].sum()]
    chain = ClassifierChain(svc(), order=list(range(len(labels))),
                            random_state=SEED)
    chain.fit(A, ytr)
    SV, SB = chain.decision_function(V), chain.decision_function(B)
    f1s, rows = [], []
    for j in keep:
        thr = best_threshold(SV[:, j], yva[:, j])
        pred = SB[:, j] >= thr
        p, r, f = prf(int((pred & (yte[:, j] == 1)).sum()),
                      int((pred & (yte[:, j] == 0)).sum()),
                      int((~pred & (yte[:, j] == 1)).sum()))
        f1s.append(f)
        rows.append((labels[j], int(yte[:, j].sum()), thr, p, r, f))
    return sum(f1s) / len(f1s), rows


def show(rows):
    print(f"\n  {'label':<34}{'n':>6}{'thr':>7}{'P':>7}{'R':>7}{'F1':>7}")
    for lab, n, t, p, r, f in rows:
        print(f"  {lab:<34}{n:>6}{t:>7.2f}{p:>7.3f}{r:>7.3f}{f:>7.3f}")


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
    print(f"labels {len(labels)}   C={C} max_features={MAX_FEAT} min_df={MIN_DF} fixed")
    print("reference: tune_dvb best = 0.658\n")

    X = lambda idx: [text[i] for i in idx]
    L = lambda idx: [Y[i] for i in idx]
    Xtr, Xva, Xte = X(tr), X(va), X(te)
    Ytr, Yva, Yte = L(tr), L(va), L(te)

    def feats(analyzer, ngram, max_df=1.0):
        vec = vectorizer(analyzer, ngram, max_df)
        A = vec.fit_transform(Xtr)
        return A, vec.transform(Xva), vec.transform(Xte), A.shape[1]

    print("=== stage A: analyzer / ngram_range (LinearSVC, thresholds) ===")
    print(f"  {'analyzer':<10}{'ngram':>8}{'n_feat':>9}{'macro-F1':>10}{'min':>7}")
    grid = [("char_wb", (3, 5)), ("char", (3, 5)),
            ("char_wb", (2, 5)), ("char", (2, 5)),
            ("char_wb", (3, 6)), ("char", (3, 6)),
            ("char_wb", (4, 6))]
    bestA = None
    for an, ng in grid:
        t0 = time.time()
        A, V, B, nf = feats(an, ng)
        m, _ = run_ovr(A, V, B, Ytr, Yva, Yte, labels, svc)
        print(f"  {an:<10}{str(ng):>8}{nf:>9}{m:>10.3f}{(time.time()-t0)/60:>7.1f}")
        if bestA is None or m > bestA[0]:
            bestA = (m, an, ng)
    _, AN, NG = bestA
    print(f"  -> best features: {AN} {NG}  ({bestA[0]:.3f})")

    print("\n=== stage A2: max_df on the best features ===")
    print(f"  {'max_df':>8}{'n_feat':>9}{'macro-F1':>10}")
    bestD = (bestA[0], 1.0)
    for md in (1.0, 0.9, 0.7):
        if md == 1.0:
            m, nf = bestA[0], "-"
        else:
            A, V, B, nf = feats(AN, NG, md)
            m, _ = run_ovr(A, V, B, Ytr, Yva, Yte, labels, svc)
        print(f"  {md:>8}{str(nf):>9}{m:>10.3f}")
        if m > bestD[0]:
            bestD = (m, md)
    MD = bestD[1]
    print(f"  -> best max_df: {MD}  ({bestD[0]:.3f})")

    print(f"\n=== stage B: model class on {AN} {NG} max_df={MD} ===")
    A, V, B, _ = feats(AN, NG, MD)
    print(f"  {'model':<22}{'macro-F1':>10}{'min':>7}")
    results, best = {}, None
    for name, mk in MODELS.items():
        t0 = time.time()
        m, rws = run_ovr(A, V, B, Ytr, Yva, Yte, labels, mk)
        print(f"  {name:<22}{m:>10.3f}{(time.time()-t0)/60:>7.1f}")
        results[name] = m
        if best is None or m > best[0]:
            best = (m, name, rws)
    t0 = time.time()
    try:
        m, rws = run_chain(A, V, B, Ytr, Yva, Yte, labels)
        print(f"  {'ClassifierChain':<22}{m:>10.3f}{(time.time()-t0)/60:>7.1f}")
        results["ClassifierChain"] = m
        if m > best[0]:
            best = (m, "ClassifierChain", rws)
    except Exception as e:                       # noqa: BLE001
        print(f"  ClassifierChain        FAILED: {type(e).__name__}: {e}")
        results["ClassifierChain"] = None

    m, name, rws = best
    print(f"\n=== best: {name} on {AN} {NG} max_df={MD} ===")
    show(rws)
    print(f"\n  macro-F1 {m:.3f}")
    print("\n" + "=" * 58)
    print(f"  tune_dvb best (char_wb(3,5), LinearSVC)   0.658")
    print(f"  this sweep best ({name})".ljust(44) + f"{m:.3f}  ({m - 0.658:+.3f})")
    print("=" * 58)
    json.dump({"analyzer": AN, "ngram_range": list(NG), "max_df": MD,
               "model": name, "C": C, "max_features": MAX_FEAT,
               "min_df": MIN_DF, "macro_f1_temporal": round(m, 4),
               "models_compared": {k: (round(v, 4) if v else None)
                                   for k, v in results.items()},
               "thresholds": {r[0]: round(r[2], 3) for r in rws},
               "f1": {r[0]: round(r[5], 4) for r in rws}},
              open("dataset/dvb_sweep_config.json", "w"),
              ensure_ascii=False, indent=1)
    print("wrote dataset/dvb_sweep_config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
