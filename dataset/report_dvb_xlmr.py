#!/usr/bin/env python3
"""Turn a finished XLM-R run into the comparison that goes in the write-up.

    .venv/bin/python dataset/report_dvb_xlmr.py ~/Downloads/dvb_xlmr_config.json

Exists because the per-label comparison has to be made against
`f1_per_label_linear_at512` -- the linear arm reading the SAME truncated text
XLM-R reads -- and not against `f1_per_label_linear_full_text`. Comparing per
label against the full-text figures understates XLM-R on every label where
truncation hurt the baseline, which is a mistake I made once by hand. The
script picks the right baseline so nobody has to remember which is which.

It also recomputes micro and weighted F1 from the per-label P/R, for runs
produced by a notebook revision that only stored macro.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVID = "COVID19"


def test_support():
    sup = collections.Counter()
    with gzip.open(ROOT / "dataset/dvb_xlmr_data.jsonl.gz", "rt") as fh:
        for line in fh:
            d = json.loads(line)
            if d["split"] == "test":
                for l in d["labels"]:
                    sup[l] += 1
    return sup


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, help="dvb_xlmr_config.json from the run")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "dataset/dvb_xlmr_result.json")
    args = ap.parse_args()

    run = json.load(open(args.config))
    meta = json.load(open(ROOT / "dataset/dvb_xlmr_meta.json"))
    base = meta["f1_per_label_linear_at512"]          # the ONLY valid baseline
    full = meta["f1_per_label_linear_full_text"]
    tgt = meta["linear_arm"]["macro_f1_temporal_truncated_512"]
    x = run["f1"]
    sup = test_support()

    missing = set(x) - set(base)
    if missing:
        sys.exit("labels in the run that the baseline lacks: %s" % sorted(missing))

    print("model %s   epoch %s of %s   device %s"
          % (run.get("model"), run.get("epoch"), run.get("epochs_planned"),
             run.get("device")))
    print("macro-F1 by epoch (test):", run.get("macro_f1_by_epoch"))
    if run.get("val_macro_f1_by_epoch"):
        print("macro-F1 by epoch (val) :", run["val_macro_f1_by_epoch"],
              "   epoch selected on", run.get("epoch_selected_on", "?"))
    else:
        print("!! this run predates val-based epoch selection; the reported "
              "epoch was chosen on test")
    print()

    print("%-34s %8s %8s %8s   %9s" % ("label", "lin@512", "xlmr", "delta",
                                       "(vs full)"))
    for k, v in sorted(x.items(), key=lambda kv: -(kv[1] - base[kv[0]])):
        print("%-34s %8.3f %8.3f %+8.3f   %+9.3f"
              % (k, base[k], v, v - base[k], v - full[k]))

    ex = [k for k in x if k != COVID]
    m19 = st.mean(x.values())
    b19 = st.mean(base[k] for k in x)
    m18 = st.mean(x[k] for k in ex)
    b18 = st.mean(base[k] for k in ex)
    wins = sum(1 for k in x if x[k] > base[k])

    # micro/weighted from the per-label table when the run did not store them
    micro = run.get("micro_f1_temporal")
    weighted = run.get("weighted_f1_temporal")
    if weighted is None:
        n = sum(sup[k] for k in x)
        weighted = sum(x[k] * sup[k] for k in x) / n

    out = {"source": str(args.config), "model": run.get("model"),
           "epoch": run.get("epoch"), "macro_f1_19": round(m19, 4),
           "macro_f1_18_excl_covid": round(m18, 4),
           "micro_f1": micro, "weighted_f1": round(weighted, 4),
           "linear_at512_macro_19": round(b19, 4),
           "linear_at512_macro_18": round(b18, 4),
           "delta_19": round(m19 - b19, 4), "delta_18": round(m18 - b18, 4),
           "labels_won": wins, "labels_total": len(x),
           "f1": x, "baseline_f1_at512": {k: base[k] for k in x}}

    print()
    print("=" * 66)
    print("  macro, all 19        linear %.4f   xlmr %.4f   %+.4f"
          % (b19, m19, m19 - b19))
    print("  macro, excl COVID19  linear %.4f   xlmr %.4f   %+.4f"
          % (b18, m18, m18 - b18))
    if micro is not None:
        print("  micro-F1             xlmr %.4f   (linear 0.804 full text)"
              % micro)
    print("  weighted-F1          xlmr %.4f   (linear 0.799 full text)"
          % weighted)
    print("  labels won           %d of %d" % (wins, len(x)))
    print("  target (linear @512) %.3f" % tgt)
    print("=" * 66)
    if micro is None:
        print("  micro-F1 needs per-label TP/FP/FN, which this run did not\n"
              "  store. weighted-F1 above is computed from test support.")

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
