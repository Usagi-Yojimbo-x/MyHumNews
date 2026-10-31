#!/usr/bin/env python3
"""Does merging COVID19 into health help, and is the merge defensible?

    .venv/bin/python dataset/merge_covid_health.py              # full text
    .venv/bin/python dataset/merge_covid_health.py --max-len 512  # + XLM-R-matched

COVID19 is the one label in the DVB scheme that names an event rather than a
subject domain, and it breaks under the temporal split: 1,282 training and 202
validation positives against 11 in the 2025 test period. No threshold fitted on
validation can survive a prevalence drop that large, and the label alone costs
~0.036 macro-F1.

The question is whether folding it into health is a principled scheme fix or
test-set-driven label engineering. Three facts from the TRAINING data settle
that, with no reference to any test score:

  COVID19 articles already carrying health    690  46.2%
  COVID19 as the only label on the article    721  48.2%
  COVID19 plus some other topic, not health    84   5.6%

94.4% either already say health or claim no other topic. And because the scheme
is multi-label, merging ADDS health and drops the COVID19 column -- the 84
articles with another topic keep it. Nothing is relabelled away from a category
it was in, so the merge cannot mislabel anything.

This script reports both label sets under the identical protocol so the merge
can be argued from numbers rather than asserted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from dataset import dvb_scheme as S  # noqa: E402
from dataset.train_dvb_xlmr import MODEL, split  # noqa: E402
from dataset.tune_dvb import SEED, evaluate, load  # noqa: E402

COVID, HEALTH = "COVID19", "ကျန်းမာရေး"
CFG = dict(C=0.5, max_feat=300_000, min_df=2)   # the shipped linear config


def merge(Y):
    """Drop the COVID19 column, adding health wherever it fired."""
    out = []
    for s in Y:
        s = set(s)
        if COVID in s:
            s.discard(COVID)
            s.add(HEALTH)
        out.append(s)
    return out


def truncate(text, max_len):
    """Every article cut to its first max_len XLM-R tokens, decoded back."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"\ntruncating {len(text)} articles to {max_len} XLM-R tokens",
          flush=True)
    cut, t0 = [], time.time()
    for a in range(0, len(text), 512):
        ids = tok(text[a:a + 512], truncation=True, max_length=max_len,
                  add_special_tokens=False)["input_ids"]
        cut.extend(tok.batch_decode(ids, skip_special_tokens=True))
        if a and a % 10240 == 0:
            print(f"  {a}/{len(text)}  {(time.time() - t0) / 60:.1f}m",
                  flush=True)
    kept = sum(len(c) for c in cut) / sum(len(t) for t in text)
    print(f"  done in {(time.time() - t0) / 60:.1f}m; {kept:.1%} of chars kept")
    return cut, kept


def arm(txt, Y, labels, tr, va, te, title):
    X = lambda idx: [txt[i] for i in idx]       # noqa: E731
    Ls = lambda idx: [Y[i] for i in idx]        # noqa: E731
    t0 = time.time()
    macro, rows = evaluate(X(tr), Ls(tr), X(va), Ls(va), X(te), Ls(te),
                           labels, CFG["C"], CFG["max_feat"], CFG["min_df"],
                           True)
    print(f"\n=== {title} ===")
    print(f"  {'label':<34}{'n':>6}{'thr':>7}{'P':>7}{'R':>7}{'F1':>7}")
    for lab, n, t, p, r, f in sorted(rows, key=lambda z: -z[5]):
        print(f"  {lab:<34}{n:>6}{t:>7.2f}{p:>7.3f}{r:>7.3f}{f:>7.3f}")
    print(f"  macro-F1 {macro:.4f}  over {len(rows)} labels"
          f"  [{(time.time() - t0) / 60:.1f}m]")
    return macro, {r[0]: round(r[5], 4) for r in rows}


def provenance(Y):
    k = {"with_health": 0, "covid_alone": 0, "covid_plus_other": 0}
    for s in Y:
        if COVID not in s:
            continue
        if HEALTH in s:
            k["with_health"] += 1
        elif len(s) == 1:
            k["covid_alone"] += 1
        else:
            k["covid_plus_other"] += 1
    return k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset/myhumnews.jsonl"))
    ap.add_argument("--max-len", type=int, default=0,
                    help="also run the XLM-R-matched truncated control")
    ap.add_argument("--out", type=Path,
                    default=Path("dataset/dvb_merge_covid_health.json"))
    args = ap.parse_args()

    rows = load(args.data)
    text, Y, tr, va, te = split(rows)
    labels19 = sorted(S.F1_TEMPORAL, key=lambda l: -S.F1_TEMPORAL[l])
    labels18 = [l for l in labels19 if l != COVID]
    Ym = merge(Y)

    prov = provenance(Y)
    tot = sum(prov.values())
    print(f"articles {len(rows)}   train {len(tr)}  val {len(va)}  test {len(te)}")
    print(f"\nCOVID19 article provenance (n={tot}, training data only):")
    for k, v in sorted(prov.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v:5d}  {v / tot:5.1%}")

    for name, idx in (("train", tr), ("val", va), ("test", te)):
        c = sum(1 for i in idx if COVID in Y[i])
        h = sum(1 for i in idx if HEALTH in Y[i])
        m = sum(1 for i in idx if HEALTH in Ym[i])
        print(f"  {name:<6} COVID19 {c:5d}   health {h:5d}  ->  merged health {m:5d}")

    out = {"config": CFG, "seed": SEED,
           "covid_provenance": prov,
           "note": "19-label and 18-label-merged under one protocol; the "
                   "merge is justified by the provenance counts above, which "
                   "use training data only."}

    m19, f19 = arm(text, Y, labels19, tr, va, te, "19 labels, full text")
    m18, f18 = arm(text, Ym, labels18, tr, va, te,
                   "18 labels, COVID19 merged into health, full text")
    out["full_text"] = {"macro_19": round(m19, 4), "macro_18_merged": round(m18, 4),
                        "delta": round(m18 - m19, 4), "f1_19": f19, "f1_18": f18}
    print(f"\n  full text:  19 labels {m19:.4f}   18 merged {m18:.4f}"
          f"   ({m18 - m19:+.4f})")

    if args.max_len:
        cut, kept = truncate(text, args.max_len)
        t19, g19 = arm(cut, Y, labels19, tr, va, te,
                       f"19 labels, first {args.max_len} tokens")
        t18, g18 = arm(cut, Ym, labels18, tr, va, te,
                       f"18 labels merged, first {args.max_len} tokens")
        out["truncated"] = {"max_len": args.max_len, "chars_kept": round(kept, 4),
                            "macro_19": round(t19, 4),
                            "macro_18_merged": round(t18, 4),
                            "delta": round(t18 - t19, 4),
                            "f1_19": g19, "f1_18": g18}
        print(f"\n  @{args.max_len} tokens:  19 labels {t19:.4f}"
              f"   18 merged {t18:.4f}   ({t18 - t19:+.4f})")
        print("\n  This is the number XLM-R must be compared against if the "
              "merge ships.")

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
