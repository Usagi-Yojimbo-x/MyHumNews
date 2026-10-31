#!/usr/bin/env python3
"""
TF-IDF on exactly the text XLM-R sees. The matched-input control.

    python -m classifier.tfidf_matched_control

WHY THIS EXISTS
---------------
XLM-R's position embeddings cap at 512 tokens, and the tokenizer probe shows
that on this corpus 512 tokens truncates **60.1% of articles and discards 29.8%
of all tokens**. The TF-IDF arm reads whole articles.

So a straight TF-IDF-vs-XLM-R comparison confounds two different things:

  1. linear char n-grams vs a fine-tuned multilingual transformer
  2. the whole article vs its first ~512 XLM-R tokens

This script removes (2) by decoding each article back from its truncated token
ids and training the SAME TF-IDF pipeline on that text. Three numbers result:

  TF-IDF, full text        what the system would actually ship
  TF-IDF, truncated text   the fair comparison point for XLM-R
  XLM-R,   truncated text  (train_xlmr.py)

The gap between the first two is the cost of truncation alone. The gap between
the last two is the model difference, which is the thing the proposal's
ablation is actually asking about.

Reporting XLM-R against full-text TF-IDF without this control would be the
standard way this ablation gets published wrong -- and here the confound is
large enough (30% of the text) that it could plausibly account for the whole
difference either way.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from classifier import humset, train_humset as T
from classifier.train_xlmr import MODEL, GOLD, TASK


def truncated_text(X: list[str], max_len: int) -> list[str]:
    """Decode each article back from the token ids XLM-R would actually receive."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    out = []
    for x in X:
        ids = tok(x, truncation=True, max_length=max_len,
                  add_special_tokens=False)["input_ids"]
        out.append(tok.decode(ids, skip_special_tokens=True))
    return out


def cv(X, Y, rows, C, maxf, label, Xa=None, Ya=None):
    """Xa/Ya are extra TRAINING-only articles. They never enter a test fold.

    For the truncated arm they must already be truncated the same way as X,
    or the training set mixes full and truncated text and the control stops
    controlling for truncation."""
    Xa = Xa or []
    Ya = Ya or []
    sig = ["|".join(sorted(y)) for y in Y]
    c = collections.Counter(sig)
    sig = [s if c[s] >= 5 else "RARE" for s in sig]
    raw = [None] * len(X)
    sc = [None] * len(X)
    skf = StratifiedKFold(5, shuffle=True, random_state=T.SEED)
    for tr, te in skf.split(X, sig):
        r, s = T.fit_predict([X[i] for i in tr] + Xa,
                             [Y[i] for i in tr] + Ya,
                             [X[i] for i in te], maxf, C)
        for j, i in enumerate(te):
            raw[i], sc[i] = r[j], s[j]
    rep = [T.repair(s, r) for s, r in zip(sc, raw)]
    return T.report(Y, rep, rows, label)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("-C", type=float, default=0.5)
    ap.add_argument("--max-features", type=int, default=300_000)
    ap.add_argument("--augment", type=Path,
                    help="extra LABELLED articles appended to every CV fold's "
                         "TRAINING split only; test folds stay pure gold")
    ap.add_argument("--augment-task", type=Path,
                    help="text source for --augment (it carries no bodies)")
    ap.add_argument("--augment-split", default="train",
                    help="keep only --augment rows with this `split` value; "
                         "empty string uses all of them")
    args = ap.parse_args()

    X, Y, rows = T.build(GOLD, TASK)
    print(f"gold {len(X)} articles")
    print(f"decoding to the first {args.max_len} XLM-R tokens ...", flush=True)
    Xt = truncated_text(X, args.max_len)
    kept = sum(len(a) for a in Xt) / sum(len(a) for a in X)
    print(f"  characters retained: {kept:.1%}  "
          f"(mean {sum(len(a) for a in X) / len(X):,.0f} -> "
          f"{sum(len(a) for a in Xt) / len(Xt):,.0f} chars)")

    Xa: list[str] = []
    Xat: list[str] = []
    Ya: list[set[str]] = []
    if args.augment:
        Xa, Ya, arows = T.build(args.augment, args.augment_task or args.augment,
                                split=args.augment_split or None)
        overlap = {r["url"] for r in arows} & {r["url"] for r in rows}
        if overlap:
            sys.exit(f"--augment leaks into the test set: {len(overlap)} shared urls")
        print(f"augment +{len(Xa)} articles into every training fold "
              f"(split={args.augment_split or 'all'}, 0 overlap with gold)")
        print(f"  truncating the augment set to {args.max_len} tokens too ...",
              flush=True)
        Xat = truncated_text(Xa, args.max_len)

    full = cv(X, Y, rows, args.C, args.max_features, "TF-IDF — FULL text",
              Xa=Xa, Ya=Ya)
    trunc = cv(Xt, Y, rows, args.C, args.max_features,
               f"TF-IDF — TRUNCATED to {args.max_len} XLM-R tokens (matched to XLM-R)",
               Xa=Xat, Ya=Ya)

    print(f"\n{'=' * 72}")
    print(f"  TF-IDF full text        macro-F1* {full:.3f}")
    print(f"  TF-IDF truncated text   macro-F1* {trunc:.3f}")
    print(f"  cost of truncation      {trunc - full:+.3f}")
    print(f"\n  Compare XLM-R against the TRUNCATED number ({trunc:.3f}),")
    print(f"  not the full-text one -- that is the matched-input comparison.")
    print(f"{'=' * 72}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
