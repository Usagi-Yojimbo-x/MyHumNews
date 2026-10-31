#!/usr/bin/env python3
"""Predict topic_dvb19 for the 68,809 articles no publisher labelled.

    .venv/bin/python dataset/predict_topics.py --dry-run   # measure only
    .venv/bin/python dataset/predict_topics.py             # measure, then write

48.9% of the release carries publisher-assigned labels. This fills the rest
with model predictions, kept rigorously distinct from them.

## Three rules, each from a measurement already in this repo

1. TRAIN ON DVB ONLY. The scheme is DVB's desk structure. The 3,842 non-DVB
   articles that happen to carry labels follow a DIFFERENT newsroom's
   convention, so mixing them into training would blur the thing being
   learned. They are held out instead -- see rule 2.

2. CROSS-OUTLET TRANSFER IS MEASURED, NOT ASSUMED. Those same 3,842 BBC,
   Irrawaddy and Myanmar Now articles are a free test set: train on DVB,
   predict on them, compare against their own publisher's labels. It is an
   imperfect test -- absence of a label is not a confirmed negative in a
   multi-label scheme, so recall is meaningful and precision only
   approximate -- but it is the only cross-outlet evidence available without
   new annotation, and it beats shipping an unvalidated guess.

3. FOUR LABELS ARE NOT PROPAGATED, AND COVID19 IS CAPPED BY DATE.
   dataset/label_consistency.py measured how often two near-identical
   articles both carry a label: social 0.152, ethnic 0.178, women/youth
   0.250, religion 0.286, against sport's 0.986. Propagating a label the
   PUBLISHER applies 15-29% consistently manufactures noise rather than
   spreading signal. COVID19 is excluded for 2025+ articles: it has 11
   positives in that period against 1,282 in training, and the fine-tuned
   model fired on everything at precision 0.049.

## Thresholds are tuned for PRECISION, not F1

A labelling pass is not a benchmark. A false positive here is a wrong label
in a published dataset that someone will train on; a false negative is a
missing label they can see is missing. Those costs are not symmetric, so each
label's threshold is the lowest margin achieving TARGET_PRECISION on held-out
DVB data, rather than the F1-optimal cut the evaluation used.
"""
from __future__ import annotations

import argparse
import collections
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
from dataset.tune_dvb import SEED, prf  # noqa: E402

RELEASE = Path("dataset/myhumnews.jsonl")
CONSISTENCY = Path("dataset/dvb_label_consistency.json")

# Below this near-duplicate agreement the publisher is not applying the label
# consistently enough to be worth propagating. 0.40 sits in the gap between
# religion (0.286) and election (0.484).
MIN_CONSISTENCY = 0.40
TARGET_PRECISION = 0.80
CFG = dict(C=0.5, max_feat=300_000, min_df=2)


def load():
    rows = []
    for line in RELEASE.open():
        d = json.loads(line)
        txt = ((d.get("title") or "") + "\n" + (d.get("body") or "")).strip()
        if not txt:
            continue
        rows.append({"url": d["url"], "outlet": d["outlet"], "text": txt,
                     "year": (d.get("published_date") or "")[:4],
                     "labels": set(d.get("topic_dvb19") or [])})
    return rows


def propagatable():
    """Labels consistent enough to propagate, by measurement not opinion."""
    c = json.load(CONSISTENCY.open())["by_threshold"]["0.35"]["labels"]
    keep, drop = [], []
    for l in S.F1_TEMPORAL:
        agree = c.get(l, {}).get("agreement")
        if agree is None or agree < MIN_CONSISTENCY:
            drop.append((l, agree))
        else:
            keep.append(l)
    return keep, drop


def precision_threshold(margins, gold, target):
    """Lowest cut reaching `target` precision; None if unreachable."""
    order = np.argsort(-margins)
    tp = fp = 0
    best = None
    for i in order:
        if gold[i]:
            tp += 1
        else:
            fp += 1
        if tp + fp >= 20:                       # ignore the noisy head
            p = tp / (tp + fp)
            if p >= target:
                best = float(margins[i])
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="measure and report, write nothing")
    ap.add_argument("--target-precision", type=float, default=TARGET_PRECISION)
    ap.add_argument("--out", type=Path,
                    default=Path("dataset/dvb_topic_predictions.jsonl"))
    ap.add_argument("--report", type=Path,
                    default=Path("dataset/dvb_topic_predictions.json"))
    args = ap.parse_args()

    keep, drop = propagatable()
    print("propagating %d of %d labels" % (len(keep), len(S.F1_TEMPORAL)))
    print("  excluded for low publisher consistency:")
    for l, a in sorted(drop, key=lambda x: (x[1] is None, x[1])):
        print("    %-34s agreement %s" % (l, "n/a" if a is None else "%.3f" % a))

    t0 = time.time()
    rows = load()
    dvb = [r for r in rows if r["outlet"] == "dvb_burmese" and r["labels"]]
    xout = [r for r in rows if r["outlet"] != "dvb_burmese" and r["labels"]]
    todo = [r for r in rows if not r["labels"]]
    print("\nloaded %s rows in %.0fs" % (format(len(rows), ","), time.time() - t0))
    print("  DVB labelled (train)        %s" % format(len(dvb), ","))
    print("  non-DVB labelled (test)     %s" % format(len(xout), ","))
    print("  unlabelled (to predict)     %s" % format(len(todo), ","))

    tr, va = train_test_split(range(len(dvb)), test_size=0.15,
                              random_state=SEED)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=CFG["max_feat"], sublinear_tf=True,
                          min_df=CFG["min_df"])
    A = vec.fit_transform([dvb[i]["text"] for i in tr])
    V = vec.transform([dvb[i]["text"] for i in va])
    X = vec.transform([r["text"] for r in xout])
    print("  vectorised %s  [%.0fs]" % (A.shape, time.time() - t0))

    thr, report = {}, {}
    print("\n%-34s%8s%9s%9s%8s%8s" % ("label", "train", "thresh", "val-P",
                                      "xout-P", "xout-R"))
    models = {}
    for lab in keep:
        ytr = np.array([1 if lab in dvb[i]["labels"] else 0 for i in tr])
        yva = np.array([1 if lab in dvb[i]["labels"] else 0 for i in va])
        if ytr.sum() < 20:
            print("%-34s%8d   skipped (under 20 training positives)"
                  % (lab, ytr.sum()))
            continue
        clf = LinearSVC(C=CFG["C"], class_weight="balanced", max_iter=3000,
                        random_state=SEED).fit(A, ytr)
        mv = clf.decision_function(V)
        t = precision_threshold(mv, yva, args.target_precision)
        if t is None:
            print("%-34s%8d   dropped: cannot reach P=%.2f on held-out DVB"
                  % (lab, ytr.sum(), args.target_precision))
            continue
        thr[lab] = t
        models[lab] = clf
        pv = prf(int(((mv >= t) & (yva == 1)).sum()),
                 int(((mv >= t) & (yva == 0)).sum()),
                 int(((mv < t) & (yva == 1)).sum()))
        # cross-outlet: the only transfer evidence that exists
        yx = np.array([1 if lab in r["labels"] else 0 for r in xout])
        if yx.sum():
            mx = clf.decision_function(X)
            px = prf(int(((mx >= t) & (yx == 1)).sum()),
                     int(((mx >= t) & (yx == 0)).sum()),
                     int(((mx < t) & (yx == 1)).sum()))
        else:
            px = (float("nan"),) * 3
        report[lab] = {"train_pos": int(ytr.sum()), "threshold": round(t, 4),
                       "val_precision": round(pv[0], 4),
                       "val_recall": round(pv[1], 4),
                       "xout_positives": int(yx.sum()),
                       "xout_precision": None if yx.sum() == 0 else round(px[0], 4),
                       "xout_recall": None if yx.sum() == 0 else round(px[1], 4)}
        print("%-34s%8d%9.3f%9.3f%8s%8s"
              % (lab, ytr.sum(), t, pv[0],
                 "-" if yx.sum() == 0 else "%.3f" % px[0],
                 "-" if yx.sum() == 0 else "%.3f" % px[1]))

    json.dump({"target_precision": args.target_precision,
               "min_consistency": MIN_CONSISTENCY, "config": CFG,
               "trained_on": "dvb_burmese publisher-labelled only",
               "n_train": len(tr), "n_val": len(va),
               "n_crossoutlet_test": len(xout), "n_to_predict": len(todo),
               "excluded_labels": {l: a for l, a in drop},
               "per_label": report}, args.report.open("w"),
              ensure_ascii=False, indent=1)
    print("\nwrote %s" % args.report)

    if args.dry_run:
        print("dry run: no predictions written")
        return 0

    print("\npredicting %s unlabelled articles" % format(len(todo), ","))
    T = vec.transform([r["text"] for r in todo])
    margins = {l: models[l].decision_function(T) for l in thr}
    n_lab = collections.Counter()
    with args.out.open("w") as fh:
        for i, r in enumerate(todo):
            labs, marg = [], {}
            for l, t in thr.items():
                m = float(margins[l][i])
                if m < t:
                    continue
                # COVID19 is unmeasurable from 2025 on: 11 positives in that
                # period. Predicting it there is predicting a dead label.
                if l == "COVID19" and r["year"] >= "2025":
                    continue
                labs.append(l)
                marg[l] = round(m, 4)
                n_lab[l] += 1
            fh.write(json.dumps({"url": r["url"], "topic_dvb19": sorted(labs),
                                 "margins": marg}, ensure_ascii=False) + "\n")
    print("wrote %s" % args.out)
    got = sum(1 for _ in args.out.open() if json.loads(_)["topic_dvb19"])
    print("  %s of %s predicted articles got >=1 label (%.1f%%)"
          % (format(got, ","), format(len(todo), ","), 100 * got / len(todo)))
    for l, c in n_lab.most_common():
        print("    %-34s %s" % (l, format(c, ",")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
