#!/usr/bin/env python3
"""
Score multi-label HumSet predictions, or measure two annotators against
each other.

    # how good is a model?
    python -m classifier.score_multilabel --gold classifier/data/humset_gold.jsonl \
                                          --pred runs/xlmr_pred.jsonl

    # how much do two label passes disagree?
    python -m classifier.score_multilabel --gold classifier/data/annot/labels-A.jsonl \
                                          --pred classifier/data/annot/labels-B.jsonl \
                                          --agreement

    # what does the gold set alone look like?
    python -m classifier.score_multilabel --gold classifier/data/humset_gold.jsonl

Rows are matched on `url`, never on position or index, because the three
label stores (annotator files, re-labelling passes, model dumps) were written
with different index bases and matching on `i` silently mis-aligned them once
already.

WHY THESE NUMBERS AND NOT ACCURACY
----------------------------------
`accuracy` has no single meaning for multi-label output, and every intuitive
definition of it flatters the model:

  per-label P/R/F1   seven independent binary problems. This is the unit the
                     proposal asks for, and the only place a label with 16
                     examples can be seen to be unmeasurable.

  macro-F1           the headline. Unweighted mean over the seven, so
                     `information_communication` (n=6) counts as much as
                     `context` (n=186). That is the point -- it refuses to let
                     the two big labels carry the score -- but it also means
                     the headline inherits the variance of the thinnest label.
                     Reported both over all 7 and over the labels with support
                     >= MIN_SUPPORT, because those two numbers answer different
                     questions and the gap between them is itself the finding.

  micro-F1           pools all label decisions. Always higher than macro here;
                     report it next to macro, never instead of it.

  exact match        fraction of articles whose WHOLE set is right. Harsh and
                     honest: it is the number a user of the system experiences.

  Hamming loss       fraction of the 383x7 individual decisions that are wrong.
                     Looks tiny by construction (a system predicting nothing at
                     all scores ~0.22 here), so it is a diagnostic, not a
                     result.

For --agreement there is no ground truth, so precision/recall are meaningless
in the usual direction: the two passes are symmetric. It reports Jaccard,
per-label Cohen's kappa, and which label pairs the two passes trade.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

from classifier import humset

MIN_SUPPORT = 20  # below this a per-label F1 is noise, not a result


# ---------------------------------------------------------------- loading

def load(path: Path, field: str = "labels") -> dict[str, set[str]]:
    """url -> label set. Tolerates `labels`, `pred`, `predicted`."""
    out: dict[str, set[str]] = {}
    fields = (field, "labels", "pred", "predicted", "pred_labels")
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            url = row.get("url")
            if not url:
                sys.exit(f"{path}:{ln}: row has no url")
            for f in fields:
                if f in row:
                    labels = row[f]
                    break
            else:
                sys.exit(f"{path}:{ln}: no label field (tried {fields})")
            if labels is None:          # skipped / unlabelled
                continue
            if isinstance(labels, str):  # single-label file
                labels = [labels]
            unknown = set(labels) - set(humset.LABELS)
            if unknown:
                sys.exit(f"{path}:{ln}: unknown labels {sorted(unknown)}")
            if url in out:
                sys.exit(f"{path}:{ln}: duplicate url {url}")
            out[url] = set(labels)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
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


# ---------------------------------------------------------------- reports

def describe(gold: dict[str, set[str]], name: str) -> None:
    n = len(gold)
    print(f"\n=== {name}: {n} articles ===")
    freq = collections.Counter(l for s in gold.values() for l in s)
    print(f"  {'label':28} {'n':>5} {'% of set':>9}   support")
    for lab in humset.LABELS:
        v = freq[lab]
        flag = "" if v >= MIN_SUPPORT else f"THIN (<{MIN_SUPPORT})"
        print(f"  {lab:28} {v:5d} {v / n:8.1%}   {flag}")
    card = collections.Counter(len(s) for s in gold.values())
    print(f"\n  labels/article  mean {sum(len(s) for s in gold.values()) / n:.2f}   "
          + "  ".join(f"{k}→{v}" for k, v in sorted(card.items())))

    pairs = collections.Counter()
    for s in gold.values():
        for a, b in ((x, y) for i, x in enumerate(sorted(s)) for y in sorted(s)[i + 1:]):
            pairs[(a, b)] += 1
    if pairs:
        print("\n  co-occurrences:")
        for (a, b), c in pairs.most_common(8):
            print(f"    {c:4d}  {a} + {b}")

    # integrity
    bad_excl = [u for u, s in gold.items() if humset.EXCLUSIVE in s and len(s) > 1]
    empty = [u for u, s in gold.items() if not s]
    print(f"\n  integrity: empty={len(empty)}  "
          f"'{humset.EXCLUSIVE}'-not-exclusive={len(bad_excl)}")


def score(gold: dict[str, set[str]], pred: dict[str, set[str]]) -> None:
    shared = sorted(set(gold) & set(pred))
    if not shared:
        sys.exit("no urls in common between gold and pred")
    print(f"\n=== scored on {len(shared)} articles "
          f"(gold {len(gold)}, pred {len(pred)}, "
          f"gold-only {len(set(gold) - set(pred))}, "
          f"pred-only {len(set(pred) - set(gold))}) ===")

    stats = {}
    for lab in humset.LABELS:
        tp = sum(1 for u in shared if lab in gold[u] and lab in pred[u])
        fp = sum(1 for u in shared if lab not in gold[u] and lab in pred[u])
        fn = sum(1 for u in shared if lab in gold[u] and lab not in pred[u])
        stats[lab] = (tp, fp, fn, tp + fn)

    print(f"\n  {'label':28} {'P':>6} {'R':>6} {'F1':>6} {'supp':>6}  {'TP':>4} {'FP':>4} {'FN':>4}")
    for lab in humset.LABELS:
        tp, fp, fn, sup = stats[lab]
        p, r, f = prf(tp, fp, fn)
        mark = "" if sup >= MIN_SUPPORT else "  thin"
        print(f"  {lab:28} {p:6.3f} {r:6.3f} {f:6.3f} {sup:6d}  {tp:4d} {fp:4d} {fn:4d}{mark}")

    f1s = {l: prf(*stats[l][:3])[2] for l in humset.LABELS}
    measurable = [l for l in humset.LABELS if stats[l][3] >= MIN_SUPPORT]
    macro_all = sum(f1s.values()) / len(f1s)
    macro_meas = (sum(f1s[l] for l in measurable) / len(measurable)) if measurable else 0.0

    TP = sum(s[0] for s in stats.values())
    FP = sum(s[1] for s in stats.values())
    FN = sum(s[2] for s in stats.values())
    _, _, micro = prf(TP, FP, FN)

    exact = sum(1 for u in shared if gold[u] == pred[u])
    lo, hi = wilson(exact, len(shared))
    wrong_cells = sum(len(gold[u] ^ pred[u]) for u in shared)
    hamming = wrong_cells / (len(shared) * len(humset.LABELS))
    jac = sum(len(gold[u] & pred[u]) / len(gold[u] | pred[u])
              for u in shared if gold[u] | pred[u]) / len(shared)

    print(f"\n  macro-F1 (all {len(humset.LABELS)} labels)        {macro_all:.3f}")
    print(f"  macro-F1 (support >= {MIN_SUPPORT}, n={len(measurable)})      {macro_meas:.3f}"
          f"   [{', '.join(measurable)}]")
    print(f"  micro-F1                          {micro:.3f}")
    print(f"  exact match                       {exact / len(shared):.3f}"
          f"   95% CI [{lo:.3f}, {hi:.3f}]  ({exact}/{len(shared)})")
    print(f"  Jaccard (mean per article)        {jac:.3f}")
    print(f"  Hamming loss                      {hamming:.4f}"
          f"   ({wrong_cells} of {len(shared) * len(humset.LABELS)} decisions)")

    # a floor worth knowing: always predict the two commonest labels
    freq = collections.Counter(l for u in shared for l in gold[u])
    top2 = {l for l, _ in freq.most_common(2)}
    base = {l: prf(sum(1 for u in shared if l in gold[u] and l in top2),
                   sum(1 for u in shared if l not in gold[u] and l in top2),
                   sum(1 for u in shared if l in gold[u] and l not in top2))[2]
            for l in humset.LABELS}
    print(f"\n  baseline, always predict {sorted(top2)}:")
    print(f"    macro-F1 {sum(base.values()) / len(base):.3f}   "
          f"exact match {sum(1 for u in shared if gold[u] == top2) / len(shared):.3f}")

    print("\n  confusions (gold label -> what was predicted instead):")
    conf = collections.Counter()
    for u in shared:
        for g in gold[u] - pred[u]:
            for p_ in pred[u] - gold[u]:
                conf[(g, p_)] += 1
    for (g, p_), c in conf.most_common(10):
        sup = stats[g][3]
        print(f"    {c:4d} ({c / sup:5.1%} of {g:26}) {g} -> {p_}")


def agreement(a: dict[str, set[str]], b: dict[str, set[str]],
              na: str, nb: str) -> None:
    shared = sorted(set(a) & set(b))
    if not shared:
        sys.exit("no overlap between the two passes")
    n = len(shared)
    print(f"\n=== agreement on {n} double-labelled articles ===")
    print(f"  A = {na}\n  B = {nb}")

    exact = sum(1 for u in shared if a[u] == b[u])
    lo, hi = wilson(exact, n)
    jac = sum(len(a[u] & b[u]) / len(a[u] | b[u])
              for u in shared if a[u] | b[u]) / n
    print(f"\n  exact set match   {exact / n:.3f}   95% CI [{lo:.3f}, {hi:.3f}]  ({exact}/{n})")
    print(f"  Jaccard           {jac:.3f}")

    print(f"\n  {'label':28} {'A':>5} {'B':>5} {'both':>5} {'kappa':>7} {'F1(B|A)':>8}")
    kappas = []
    for lab in humset.LABELS:
        ca = sum(1 for u in shared if lab in a[u])
        cb = sum(1 for u in shared if lab in b[u])
        both = sum(1 for u in shared if lab in a[u] and lab in b[u])
        neither = sum(1 for u in shared if lab not in a[u] and lab not in b[u])
        po = (both + neither) / n
        pe = (ca / n) * (cb / n) + (1 - ca / n) * (1 - cb / n)
        k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        _, _, f = prf(both, cb - both, ca - both)
        kappas.append(k)
        print(f"  {lab:28} {ca:5d} {cb:5d} {both:5d} {k:7.3f} {f:8.3f}")
    finite = [k for k in kappas if not math.isnan(k)]
    print(f"\n  mean kappa over {len(finite)} labels   {sum(finite) / len(finite):.3f}")

    # A kappa of 0 is not necessarily disagreement. If one pass is CONSTANT for
    # a label -- which happens whenever the subset was selected on that label,
    # e.g. "re-label everything pass 1 called `other`" -- then chance agreement
    # already equals observed agreement and kappa is 0 by construction. Say so,
    # loudly, because the alternative is reading it as total disagreement.
    const = [l for l in humset.LABELS
             if sum(1 for u in shared if l in a[u]) in (0, n)]
    if const:
        print(f"\n  CAVEAT: pass A is constant for {len(const)} of {len(humset.LABELS)} labels")
        print(f"    {', '.join(const)}")
        print("    kappa is 0 or undefined for these BY CONSTRUCTION, not by")
        print("    disagreement -- a constant rater has no variance to correct")
        print("    for. This is what you get when the subset was selected on one")
        print("    of A's labels. Read the swap table below instead; for a real")
        print("    inter-annotator figure, double-label an UNSELECTED sample.")

    print("\n  what the two passes trade (A had -> B had instead):")
    swap = collections.Counter()
    for u in shared:
        for x in a[u] - b[u]:
            for y in b[u] - a[u]:
                swap[(x, y)] += 1
    for (x, y), c in swap.most_common(10):
        print(f"    {c:4d}  {x} -> {y}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--pred", type=Path,
                    help="predictions, or the second annotator's file")
    ap.add_argument("--agreement", action="store_true",
                    help="treat --pred as a second annotator, not a model")
    ap.add_argument("--field", default="labels")
    ap.add_argument("--describe", action="store_true",
                    help="also describe the gold set on its own")
    args = ap.parse_args()

    gold = load(args.gold, args.field)
    if args.describe or not args.pred:
        describe(gold, args.gold.name)
    if not args.pred:
        return 0

    pred = load(args.pred, args.field)
    if args.agreement:
        agreement(gold, pred, args.gold.name, args.pred.name)
    else:
        score(gold, pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
