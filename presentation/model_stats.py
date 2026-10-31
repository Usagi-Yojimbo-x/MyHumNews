#!/usr/bin/env python3
"""Recompute per-label P/R/F1 and macro-F1* for both arms from existing
prediction dumps and XLM-R checkpoints. No training, no GPU.
Uses classifier.train_humset's own prf/repair/MIN_SUPPORT so the protocol is
identical to the reported runs.
"""
from __future__ import annotations
import json, os, statistics as st, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from classifier import humset
from classifier import train_humset as T

L = humset.LABELS

def score(Y, P):
    """per-label P/R/F1 + macro over support>=MIN_SUPPORT, matching T.report."""
    out, macro_all, macro_sup = {}, [], []
    for l in L:
        tp = sum(1 for y, p in zip(Y, P) if l in y and l in p)
        fp = sum(1 for y, p in zip(Y, P) if l not in y and l in p)
        fn = sum(1 for y, p in zip(Y, P) if l in y and l not in p)
        pr, rc, f1 = T.prf(tp, fp, fn)
        supp = sum(1 for y in Y if l in y)
        out[l] = {"P": pr, "R": rc, "F1": f1, "supp": supp,
                  "pred": sum(1 for p in P if l in p)}
        macro_all.append(f1)
        if supp >= T.MIN_SUPPORT:
            macro_sup.append(f1)
    exact = sum(1 for y, p in zip(Y, P) if y == p) / len(Y)
    return {"per_label": out, "macro_all": st.mean(macro_all),
            "macro_sup": st.mean(macro_sup), "n_sup": len(macro_sup),
            "exact": exact}

res = {}

# ---- TF-IDF, full text, from its dump
rows = [json.loads(l) for l in open("classifier/data/humset_tfidf_cv.jsonl")]
Y = [set(r["labels"]) for r in rows]
res["tfidf_full"] = score(Y, [set(r["pred"]) for r in rows])

# ---- XLM-R run 2 (local MPS), from its dump
rows2 = [json.loads(l) for l in open("classifier/data/humset_xlmr_cv_v2.jsonl")]
res["xlmr_local_v2"] = score([set(r["labels"]) for r in rows2],
                             [set(r["pred"]) for r in rows2])

# ---- XLM-R, the three Colab seeds, from checkpoints (scores are threshold-shifted)
X, Ygold, grows = __import__("classifier.train_xlmr", fromlist=["load"]).load()
for seed in (20261015, 20261016, 20261017):
    ck = json.load(open(f"classifier/data/xlmr_ckpt_s{seed}.json"))
    assert ck["seed"] == seed and ck["n"] == len(X), (seed, ck["seed"], ck["n"])
    sc = [None] * len(X)
    for per in ck["folds"].values():
        for i, s in per.items():
            sc[int(i)] = s
    assert all(s is not None for s in sc)
    raw = [{l for l in L if s[l] > 0} for s in sc]
    rep = [T.repair(s, r) for s, r in zip(sc, raw)]
    res[f"xlmr_seed{seed}"] = score(Ygold, rep)
    ck_aug = json.load(open(f"classifier/data/xlmr_ckpt_aug_s{seed}.json"))
    assert ck_aug["seed"] == seed and ck_aug["n"] == len(X) and ck_aug.get("n_aug") == 180
    sc_a = [None] * len(X)
    for per in ck_aug["folds"].values():
        for i, s in per.items():
            sc_a[int(i)] = s
    raw_a = [{l for l in L if s[l] > 0} for s in sc_a]
    rep_a = [T.repair(s, r) for s, r in zip(sc_a, raw_a)]
    res[f"xlmr_aug_seed{seed}"] = score(Ygold, rep_a)
# the cell-12 single-seed checkpoint (same seed, different run)
ck = json.load(open("classifier/data/xlmr_ckpt.json"))
sc = [None] * len(X)
for per in ck["folds"].values():
    for i, s in per.items(): sc[int(i)] = s
raw = [{l for l in L if s[l] > 0} for s in sc]
res["xlmr_seed20261015_runA"] = score(Ygold, [T.repair(s, r) for s, r in zip(sc, raw)])

json.dump(res, open("presentation/model_stats.json", "w"), indent=1)

print(f"{'run':28s} {'macro*':>7} {'macroAll':>9} {'exact':>7}")
for k, v in res.items():
    print(f"{k:28s} {v['macro_sup']:7.3f} {v['macro_all']:9.3f} {v['exact']:7.3f}")
seeds = [res[f"xlmr_seed{s}"]["macro_sup"] for s in (20261015, 20261016, 20261017)]
print(f"\nXLM-R 3 seeds: " + "  ".join(f"{x:.3f}" for x in seeds)
      + f"   mean {st.mean(seeds):.3f}  sd {st.stdev(seeds):.3f}")
print("\nper-label F1:")
print(f"{'label':28s} {'supp':>4} {'TFIDFfull':>10} " + " ".join(f"s{s%100:02d}" for s in (20261015,20261016,20261017)) + "   mean±sd")
for l in L:
    f = [res[f"xlmr_seed{s}"]["per_label"][l]["F1"] for s in (20261015,20261016,20261017)]
    print(f"{l:28s} {res['tfidf_full']['per_label'][l]['supp']:>4} "
          f"{res['tfidf_full']['per_label'][l]['F1']:10.3f} "
          + " ".join(f"{x:.3f}" for x in f)
          + f"   {st.mean(f):.3f}±{st.stdev(f):.3f}")

# ---- floor: always predict the two most common labels
floor = score(Ygold, [{"context", "shock_event"} for _ in Ygold])
res["floor"] = floor
json.dump(res, open("presentation/model_stats.json", "w"), indent=1)
print(f"\nfloor (always context+shock_event): macro* {floor['macro_sup']:.3f} "
      f"macroAll {floor['macro_all']:.3f} exact {floor['exact']:.3f}")
