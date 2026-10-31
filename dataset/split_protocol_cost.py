#!/usr/bin/env python3
"""What does the temporal split cost, and what do the other metrics say?

    .venv/bin/python dataset/split_protocol_cost.py

Two questions that come up when these numbers are set beside published work.

1. METRIC. Multi-label text classification papers usually headline micro-F1 or
   weighted-F1. Macro-F1 gives a label with 76 training examples the same
   weight as one with 9,426, so on a long-tailed label set it reads far lower
   than micro on the identical predictions. Reporting macro alone against
   papers reporting micro is not a like-for-like comparison. This script
   reports macro, micro and weighted together, every time.

2. PROTOCOL. The headline split here is temporal: train on <=2024, test on
   2025+. Almost every comparable paper splits randomly. A temporal split is
   strictly harder, because it adds distribution shift on top of the
   classification problem -- and in this corpus the shift is severe, not
   incidental: COVID19 goes from 202 validation positives to 11 in test, and
   election goes from 96 training positives to 458 in test.

   So the random-split number is reported here as a DIAGNOSTIC, to size that
   handicap. The temporal number stays the headline, because predicting next
   year's news from this year's is the task a deployed classifier actually
   faces. Quoting the random-split figure as the result would be choosing the
   easier protocol after seeing both.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from dataset import dvb_scheme as S  # noqa: E402
from dataset.tune_dvb import SEED, best_threshold, load, prf  # noqa: E402

CFG = dict(C=0.5, max_feat=300_000, min_df=2)   # the shipped linear config


def score(Xtr, Ytr, Xva, Yva, Xte, Yte, labels):
    """Per-label P/R/F1 plus macro, micro and weighted aggregates."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=CFG["max_feat"], sublinear_tf=True,
                          min_df=CFG["min_df"])
    A, V, B = vec.fit_transform(Xtr), None, None
    V = vec.transform(Xva)
    B = vec.transform(Xte)

    rows, TP = [], 0
    FP = FN = 0
    for lab in labels:
        ytr = np.array([1 if lab in s else 0 for s in Ytr])
        yva = np.array([1 if lab in s else 0 for s in Yva])
        yte = np.array([1 if lab in s else 0 for s in Yte])
        if ytr.sum() == 0 or yte.sum() == 0:
            continue
        clf = LinearSVC(C=CFG["C"], class_weight="balanced", max_iter=3000,
                        random_state=SEED)
        clf.fit(A, ytr)
        thr = best_threshold(clf.decision_function(V), yva)
        pred = clf.decision_function(B) >= thr
        tp = int((pred & (yte == 1)).sum())
        fp = int((pred & (yte == 0)).sum())
        fn = int((~pred & (yte == 1)).sum())
        p, r, f = prf(tp, fp, fn)
        TP += tp; FP += fp; FN += fn
        rows.append((lab, int(yte.sum()), p, r, f))

    macro = sum(r[4] for r in rows) / len(rows)
    n_tot = sum(r[1] for r in rows)
    weighted = sum(r[4] * r[1] for r in rows) / n_tot
    mp, mr, micro = prf(TP, FP, FN)
    return dict(macro_f1=round(macro, 4), micro_f1=round(micro, 4),
                weighted_f1=round(weighted, 4),
                micro_precision=round(mp, 4), micro_recall=round(mr, 4),
                n_labels=len(rows), n_test_assignments=n_tot,
                per_label={r[0]: round(r[4], 4) for r in rows})


def report(name, d):
    print(f"\n=== {name} ===")
    print(f"  macro-F1     {d['macro_f1']:.4f}   over {d['n_labels']} labels")
    print(f"  micro-F1     {d['micro_f1']:.4f}   "
          f"(P {d['micro_precision']:.3f} / R {d['micro_recall']:.3f})")
    print(f"  weighted-F1  {d['weighted_f1']:.4f}   "
          f"over {d['n_test_assignments']} test assignments")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset/myhumnews.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("dataset/dvb_split_protocol_cost.json"))
    args = ap.parse_args()

    rows = load(args.data)
    labels = sorted(S.F1_TEMPORAL, key=lambda l: -S.F1_TEMPORAL[l])
    text = [(d.get("title") or "") + "\n" + (d.get("body") or "") for d, _ in rows]
    Y = [set(ls) for _, ls in rows]
    year = [(d.get("published_date") or "")[:4] for d, _ in rows]

    X = lambda i: [text[j] for j in i]        # noqa: E731
    L = lambda i: [Y[j] for j in i]           # noqa: E731
    out = {"config": CFG, "seed": SEED, "n_articles": len(rows)}

    # ---- headline: temporal ------------------------------------------
    tr_all = [i for i in range(len(text)) if year[i] and year[i] <= "2024"]
    te = [i for i in range(len(text)) if year[i] and year[i] >= "2025"]
    tr, va = train_test_split(tr_all, test_size=0.15, random_state=SEED)
    print(f"temporal   train {len(tr)}  val {len(va)}  test {len(te)}", flush=True)
    t0 = time.time()
    out["temporal"] = score(X(tr), L(tr), X(va), L(va), X(te), L(te), labels)
    report("temporal split (the headline protocol)", out["temporal"])
    print(f"  [{(time.time() - t0) / 60:.1f}m]")

    # ---- diagnostic: random, same sizes ------------------------------
    dated = [i for i in range(len(text)) if year[i]]
    r_tr_all, r_te = train_test_split(
        dated, test_size=len(te) / len(dated), random_state=SEED)
    r_tr, r_va = train_test_split(r_tr_all, test_size=0.15, random_state=SEED)
    print(f"\nrandom     train {len(r_tr)}  val {len(r_va)}  test {len(r_te)}",
          flush=True)
    t0 = time.time()
    out["random"] = score(X(r_tr), L(r_tr), X(r_va), L(r_va), X(r_te), L(r_te),
                          labels)
    report("random split (diagnostic only — NOT the headline)", out["random"])
    print(f"  [{(time.time() - t0) / 60:.1f}m]")

    d = {k: out["random"][k] - out["temporal"][k]
         for k in ("macro_f1", "micro_f1", "weighted_f1")}
    out["cost_of_temporal_split"] = {k: round(v, 4) for k, v in d.items()}
    print("\n" + "=" * 62)
    print("  what the temporal protocol costs, same model and label set")
    for k, v in d.items():
        print(f"    {k:<14} temporal {out['temporal'][k]:.4f}"
              f"   random {out['random'][k]:.4f}   ({v:+.4f})")
    print("=" * 62)
    print("  The temporal figure stays the headline. The random figure exists\n"
          "  only to say how much of the gap to published work is protocol\n"
          "  rather than model.")

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
