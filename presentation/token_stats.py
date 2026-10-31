#!/usr/bin/env python3
"""Per-article XLM-R token counts for the 383 gold articles, using the same
load path and tokenizer as classifier/train_xlmr.py. Writes token_stats.json.
"""
from __future__ import annotations
import json, os, statistics as st, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from classifier.train_xlmr import load, MODEL
from transformers import AutoTokenizer

MAX_LEN = 512
X, Y, rows = load()
tok = AutoTokenizer.from_pretrained(MODEL)
toks = [len(tok(x, truncation=False)["input_ids"]) for x in X]
chars = [len(x) for x in X]
n = len(X)
lost = sum(max(0, t - MAX_LEN) for t in toks)
over = sum(1 for t in toks if t > MAX_LEN)
out = {
    "model": MODEL, "n": n, "max_len": MAX_LEN,
    "chars_mean": st.mean(chars), "chars_median": st.median(chars),
    "tokens_mean": st.mean(toks), "tokens_median": st.median(toks),
    "tokens_p90": sorted(toks)[int(.9 * n)], "tokens_max": max(toks),
    "tokens_per_char": st.mean(toks) / st.mean(chars),
    "truncated": over, "truncated_frac": over / n,
    "tokens_total": sum(toks), "tokens_discarded": lost,
    "tokens_discarded_frac": lost / sum(toks),
    "per_article": [{"tokens": t, "chars": c, "labels": sorted(y)}
                    for t, c, y in zip(toks, chars, Y)],
}
json.dump(out, open("presentation/token_stats.json", "w"), indent=1)
for k, v in out.items():
    if k != "per_article":
        print(f"  {k:24s} {v}")
