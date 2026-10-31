#!/usr/bin/env python3
"""
Score the model against the hand labels.

    python -m classifier.score_labels

This is the first real measurement of the model on RFA. Until now RFA's 2,427
topics were predictions with no ground truth behind them, and the headline
0.856 was measured on DVB/BBC/Irrawaddy -- outlets RFA has nothing to do with.

What it reports, and why:

  accuracy + 95% CI     at n=200 the interval is roughly +/-5 points, so the
                        point estimate alone over-claims. Wilson interval,
                        which behaves sensibly near 0 and 1 unlike the normal
                        approximation.

  ACCURACY BY MARGIN    the one that matters most. The high-confidence subset
                        (outlet OR margin >= 0.5, 90% of the corpus) rests on a
                        threshold calibrated on the labelled outlets. If RFA
                        accuracy does NOT rise with margin here, that subset is
                        not trustworthy and the claim has to be withdrawn.

  per class             precision/recall/F1 where support allows. A random
                        sample leaves the tail classes thin on purpose -- rows
                        with support under 10 are marked as noise, not results.

  confusions            where RFA specifically goes wrong, which may differ
                        from the DVB-driven conflict<->politics pattern.

Skipped articles (boilerplate, non-articles) are excluded from scoring and
counted separately. They are a finding about the corpus, not model errors.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for a proportion. Correct near 0 and 1, unlike k/n +/- z*se."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(HERE, "data", "labels.jsonl"))
    ap.add_argument("--pred", default=os.path.join(HERE, "data", "predicted.jsonl"))
    ap.add_argument("--min-support", type=int, default=10)
    a = ap.parse_args()

    if not os.path.exists(a.labels):
        print("no labels yet -- run: python -m classifier.label_cli",
              file=sys.stderr)
        return 1

    pred = {}
    with open(a.pred, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                pred[r["url"]] = r

    rows, skipped, ambiguous, missing = [], 0, 0, 0
    with open(a.labels, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r["label"]:
                if r.get("reason") == "ambiguous":
                    ambiguous += 1
                else:
                    skipped += 1
                continue
            p = pred.get(r["url"])
            if p is None:
                missing += 1
                continue
            rows.append((r["label"], p["topic"], p["margin"], p["site"],
                         p.get("outlet_label")))

    if not rows:
        print("nothing scorable yet", file=sys.stderr)
        return 1

    site = collections.Counter(r[3] for r in rows).most_common(1)[0][0]
    has_outlet = all(r[4] for r in rows)
    n = len(rows)
    correct = sum(1 for t, p, _, _, _ in rows if t == p)
    acc = correct / n
    lo, hi = wilson(correct, n)

    print(f"=== {site}: model vs hand labels ===")
    print(f"  scored      {n}")
    if skipped:
        print(f"  skipped     {skipped}  (not articles; excluded)")
    if ambiguous:
        print(f"  ambiguous   {ambiguous}  (taxonomy does not decide them)")
    if missing:
        print(f"  no prediction {missing}")
    print(f"  accuracy    {acc:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")

    # Ambiguous articles are real articles the taxonomy cannot decide. Dropping
    # them silently flatters the score, so report the bound instead: best case
    # the model got them all right, worst case all wrong. The truth is inside.
    if ambiguous:
        n_all = n + ambiguous
        best = (correct + ambiguous) / n_all
        worst = correct / n_all
        print(f"\n  counting the {ambiguous} ambiguous articles:")
        print(f"    best case (all correct)  {best:.3f}")
        print(f"    worst case (all wrong)   {worst:.3f}")
        print(f"    -> true accuracy is in [{worst:.3f}, {best:.3f}]")
        frac = ambiguous / n_all
        if frac > 0.10:
            print(f"    {frac:.0%} ambiguous is HIGH -- the taxonomy needs work "
                  f"before these numbers mean much")
        else:
            print(f"    {frac:.0%} ambiguous: narrow enough that the headline "
                  f"figure stands")

    labels = sorted({r[0] for r in rows} | {r[1] for r in rows})
    f1s, shown = [], 0
    print(f"\n=== per class ===")
    print(f"  {'topic':14}{'prec':>7}{'rec':>7}{'F1':>7}{'support':>9}")
    for c in labels:
        tp = sum(1 for t, p, _, _, _ in rows if t == c and p == c)
        fp = sum(1 for t, p, _, _, _ in rows if t != c and p == c)
        fn = sum(1 for t, p, _, _, _ in rows if t == c and p != c)
        sup = tp + fn
        p_, r_, f_ = prf(tp, fp, fn)
        flag = "" if sup >= a.min_support else "   <- thin, noise not result"
        print(f"  {c:14}{p_:>7.3f}{r_:>7.3f}{f_:>7.3f}{sup:>9}{flag}")
        if sup >= a.min_support:
            f1s.append(f_)
            shown += 1
    if f1s:
        print(f"\n  macro-F1 over the {shown} classes with support >= "
              f"{a.min_support}: {sum(f1s)/len(f1s):.3f}")
        print(f"  (not comparable to the corpus macro-F1 of 0.796, which "
              f"averages all 10)")

    if has_outlet:
        # The measurement this sample exists for. outlet_lab = the outlet's own
        # section name, mapped by taxonomy.py. If it rarely matches a human
        # reading, then held-out accuracy is the model reproducing a noisy
        # label source, and every figure resting on it must be restated.
        gh = sum(1 for t, _, _, _, g in rows if t == g)
        mg = sum(1 for _, p, _, _, g in rows if p == g)
        glo, ghi = wilson(gh, n)
        mlo, mhi = wilson(mg, n)
        print(f"\n=== THE CIRCULARITY TEST ===")
        print(f"  model vs hand   {acc:.3f}  [{lo:.3f}, {hi:.3f}]   "
              f"<- the model's real accuracy")
        print(f"  outlet vs hand  {gh/n:.3f}  [{glo:.3f}, {ghi:.3f}]   "
              f"<- how good {site}'s OWN section labels are")
        print(f"  model vs outlet {mg/n:.3f}  [{mlo:.3f}, {mhi:.3f}]   "
              f"<- the number usually quoted")
        print()
        if gh / n >= 0.85:
            print(f"  the outlet label agrees with a human {gh/n:.0%} of the time: "
                  f"{site}'s section names are a")
            print(f"  sound proxy for topic, so the held-out figure means what "
                  f"it appears to mean.")
        elif gh / n >= 0.70:
            print(f"  the outlet label agrees with a human only {gh/n:.0%} of the time. "
                  f"{site}'s labels carry real")
            print(f"  noise, so held-out accuracy is an optimistic ceiling, "
                  f"not a measurement.")
        else:
            print(f"  the outlet label agrees with a human just {gh/n:.0%} of the time. "
                  f"The held-out figure is")
            print(f"  largely agreement with a NOISY label source and should "
                  f"not be quoted as accuracy.")
        print()
        print(f"  ceiling: a model that reproduced the outlet label perfectly would score "
              f"{gh/n:.3f} against a human.")
        if acc <= gh / n:
            print(f"  the model is at {acc/(gh/n)*100:.0f}% of that ceiling.")
        else:
            print(f"  the model BEATS the outlet label against human judgement "
                  f"({acc:.3f} > {gh/n:.3f}) -- it has")
            print(f"  learned the topic better than the label source it was "
                  f"trained on.")

        three = collections.Counter()
        for t, p, _, _, g in rows:
            three[(p == t, g == t)] += 1
        print(f"\n  {'':22}{'outlet right':>14}{'outlet wrong':>14}")
        for mr in (True, False):
            lbl = "model right" if mr else "model wrong"
            print(f"  {lbl:22}{three[(mr, True)]:>14}{three[(mr, False)]:>14}")
        both_wrong = three[(False, False)]
        print(f"\n  both wrong on {both_wrong} ({both_wrong/n:.0%}): these are "
              f"where the taxonomy itself")
        print(f"  may be the problem, not the model.")

    print(f"\n=== ACCURACY BY MARGIN -- does the threshold transfer? ===")
    print(f"  {'margin>=':>9}{'accuracy':>10}{'n':>6}{'coverage':>10}{'95% CI':>18}")
    for thr in (0.0, 0.3, 0.5, 0.75, 1.0, 1.5):
        sub = [(t, p) for t, p, m, _, _ in rows if m >= thr]
        if not sub:
            continue
        k = sum(1 for t, p in sub if t == p)
        l2, h2 = wilson(k, len(sub))
        print(f"  {thr:>9.2f}{k/len(sub):>10.3f}{len(sub):>6}"
              f"{len(sub)/n:>9.0%}   [{l2:.3f}, {h2:.3f}]")

    bands = [("low  <0.3", 0.0, 0.3), ("mid  0.3-1.0", 0.3, 1.0),
             ("high >=1.0", 1.0, 99.0)]
    print(f"\n  disjoint bands (the ordering test):")
    for name, lo_, hi_ in bands:
        sub = [(t, p) for t, p, m, _, _ in rows if lo_ <= m < hi_]
        if sub:
            k = sum(1 for t, p in sub if t == p)
            print(f"    {name:14}{k/len(sub):>7.3f}  n={len(sub)}")

    print(f"\n=== biggest confusions (true -> predicted) ===")
    conf = collections.Counter((t, p) for t, p, _, _, _ in rows if t != p)
    if not conf:
        print("  none")
    for (t, p), k in conf.most_common(10):
        sup = sum(1 for x, _, _, _, _ in rows if x == t)
        print(f"  {k:>4} ({k/sup*100:4.1f}% of {t})  {t} -> {p}")

    print(f"\n=== context ===")
    print(f"  held-out accuracy on labelled outlets: dvb 0.875, "
          f"irrawaddy 0.814, bbc 0.663")
    print(f"  {site} measured here:                   {acc:.3f} "
          f"[{lo:.3f}, {hi:.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
