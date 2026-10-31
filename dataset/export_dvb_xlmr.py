#!/usr/bin/env python3
"""Export the DVB task as a flat file for GPU training off this machine.

    .venv/bin/python -u -m dataset.export_dvb_xlmr

Writes dataset/dvb_xlmr_data.jsonl.gz  -- one row per article:
    {"split": "train"|"val"|"test", "text": "...", "labels": [...]}
and dataset/dvb_xlmr_meta.json with the label order and the linear-arm
reference scores.

WHY THE SPLIT IS RESOLVED HERE AND NOT IN THE NOTEBOOK
------------------------------------------------------
The whole value of the XLM-R number is that it is comparable to the linear
arm's 0.658. That holds only if the labels, the temporal split and the
validation carve-out are byte-identical. All of that logic lives in
dvb_scheme.py and tune_dvb.py and is already tested, so it runs HERE, once,
and ships as a plain "split" field. A notebook that re-derived the split from
a seed would be one library-version change away from a silently different
partition and a meaningless comparison.

CHARACTER TRUNCATION
--------------------
XLM-R cannot read past 512 tokens and Burmese runs ~0.306 tokens per
character, so 512 tokens is ~1,670 characters on average. Exporting whole
articles would trade a much larger upload for text the model never sees.
--max-chars keeps a generous margin over that (the default is ~800 tokens'
worth even for unusually dense text) so truncation here cannot bite before
the tokenizer's own 512 cut does. The linear arm's 0.658 was measured on
WHOLE articles, which is exactly why train_dvb_xlmr.py --linear-control
exists; that control runs locally on the full text, not on this export.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import dvb_scheme as S                      # noqa: E402
from dataset.tune_dvb import SEED, load                   # noqa: E402
from dataset.train_dvb_xlmr import split                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset/myhumnews.jsonl"))
    ap.add_argument("--max-chars", type=int, default=2600,
                    help="per-article character cap; must stay comfortably "
                         "above what 512 XLM-R tokens can cover")
    ap.add_argument("--out", type=Path,
                    default=Path("dataset/dvb_xlmr_data.jsonl.gz"))
    args = ap.parse_args()

    rows = load(args.data)
    labels = sorted(S.F1_TEMPORAL, key=lambda l: -S.F1_TEMPORAL[l])
    text, Y, tr, va, te = split(rows)

    where = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        for i in idx:
            where[i] = name

    keep = set(labels)
    n, cut = 0, 0
    with gzip.open(args.out, "wt", encoding="utf-8") as fh:
        for i in sorted(where):
            t = text[i]
            if len(t) > args.max_chars:
                t = t[:args.max_chars]
                cut += 1
            fh.write(json.dumps({"split": where[i], "text": t,
                                 "labels": sorted(Y[i] & keep)},
                                ensure_ascii=False) + "\n")
            n += 1

    meta = {"labels": labels,
            "n_rows": n,
            "splits": {"train": len(tr), "val": len(va), "test": len(te)},
            "max_chars": args.max_chars,
            "truncated_at_export": cut,
            "seed": SEED,
            "linear_arm": {
                "macro_f1_temporal_full_text": 0.658,
                "model": "TF-IDF char_wb(3,5) 300k min_df=2 + one-vs-rest "
                         "LinearSVC C=0.5 class_weight=balanced, per-label "
                         "thresholds fitted on val",
                "note": "measured on WHOLE articles; see "
                        "train_dvb_xlmr.py --linear-control for the "
                        "truncation-matched figure"},
            # TWO per-label baselines, and they are not interchangeable.
            # XLM-R must be compared against the _at512 one: it is the same
            # linear model scored on the same truncated text XLM-R reads.
            # Comparing per label against the full-text dict understates
            # XLM-R on every label where truncation hurt the linear arm.
            "f1_per_label_linear_full_text": S.F1_TEMPORAL,
            "f1_per_label_linear_at512": json.load(
                open("dataset/dvb_linear_control.json"))["f1"]}
    json.dump(meta, open("dataset/dvb_xlmr_meta.json", "w"),
              ensure_ascii=False, indent=1)

    mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out}  {mb:.1f} MB gzipped")
    print(f"  rows {n}   train {len(tr)}  val {len(va)}  test {len(te)}")
    print(f"  labels {len(labels)}")
    print(f"  character-truncated at export: {cut} ({cut/n:.1%})")
    print("wrote dataset/dvb_xlmr_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
