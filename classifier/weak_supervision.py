#!/usr/bin/env python3
"""
Free corpus-scale weak supervision from the outlets' own section labels.

    python -m classifier.weak_supervision --audit      # what the rules buy, vs gold
    python -m classifier.weak_supervision --build      # write the weak pool
    python -m classifier.weak_supervision --evaluate   # gold-only vs gold+weak, per label

This is the ZERO-COST substitute for the blocked LLM pre-labelling step
(`prelabel_llm.py`, ~$424 for the corpus). Categories cover all 38,152
articles and cost nothing.

IT IS A PARTIAL SUBSTITUTE, AND THE SHAPE OF THE GAP MATTERS
------------------------------------------------------------
Measured on the 383 gold articles, only three labels get a usable rule:

  sport|football       -> {other}         27/27  = 100%   COMPLETE label set
  politics             -> context         22/24  =  92%   positive only
  peace/military       -> shock_event     30/33  =  91%   positive only

and `displacement`, `casualties`, `humanitarian_access` and
`information_communication` get NOTHING. The outlets have no section that
corresponds to them.

So weak supervision helps the three labels that are already strongest
(0.756 / 0.787 / 0.867) and does nothing for the four that are weak or
unmeasurable. That is a real limit, not a tuning problem: it follows from what
a newsroom's section list is for. The two zero-F1 labels can only be fixed by
labelling more examples of them.

The `other` rule is the valuable one, because `other` is EXCLUSIVE: a sport
article's full label set is known to be exactly {other}. Those 2,966 articles
are therefore complete supervision -- positive for `other` and negative for all
six others -- not just a single positive.

PARTIAL LABELS AND ONE-VS-REST
------------------------------
A weak rule asserts something about ONE label and says nothing about the rest.
Training one-vs-rest on such rows as if the unasserted labels were negative
would inject false negatives at scale.

So supervision is tracked per label. Each row carries `pos` (labels asserted
present) and `neg` (labels asserted absent); for label L, only rows that
assert L in one direction or the other are used to train L's classifier. Gold
rows assert all seven. This is the only honest way to mix the two sources, and
it is why this is a separate script rather than a flag on `train_humset.py`.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from classifier import humset, train_humset as T

CORPUS = Path("data-vm")
GOLD = Path("classifier/data/humset_gold.jsonl")
TASK = Path("classifier/data/humset_task.jsonl")
OUT = Path("classifier/data/weak_pool.jsonl")

ALL = set(humset.LABELS)

# Burmese section names, with the English gloss. Each rule states what it
# asserts PRESENT and what it asserts ABSENT -- never "all the rest are absent"
# unless the label is `other`, which is exclusive and so does imply exactly that.
SPORT = {"အားကစား", "ဘောလုံး"}                       # sport, football
POLITICS = {"နိုင်ငံရေး", "Politics"}                  # politics
MILITARY = {"ငြိမ်းချမ်းရေး/စစ်ရေး", "မြန်မာပြည်တွင်းစစ်"}  # peace+war, civil war
WEATHER = {"သဘာဝပတ်ဝန်းကျင်/ရာသီဥတု/မိုးလေဝသ"}        # environment/climate/weather
CRIME = {"မှုခင်း", "မှုခင်း/မူးယစ်ဆေး"}                # crime, crime/drugs
ECONOMY = {"စီးပွားရေး"}                              # economy

RULES = [
    # name,                      matches,                 pos,              neg
    ("sport -> other (exclusive)", SPORT, {"other"}, ALL - {"other"}),
    ("politics -> context",        POLITICS, {"context"}, {"other"}),
    ("military -> shock_event",    MILITARY, {"shock_event"}, {"other"}),
    ("weather -> shock_event",     WEATHER, {"shock_event"}, {"other"}),
    ("crime -> shock_event",       CRIME, {"shock_event"}, {"other"}),
    ("economy -> context",         ECONOMY, {"context"}, {"other"}),
]


def load_corpus() -> list[dict]:
    rows, seen = [], set()
    for f in sorted(CORPUS.glob("*/parsed/articles.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            u, b = r.get("url"), (r.get("body") or "")
            if not u or u in seen or len(b) < 200:
                continue
            seen.add(u)
            c = r.get("categories") or []
            if isinstance(c, str):
                c = [c]
            rows.append({"url": u, "site": r.get("outlet"),
                         "title": r.get("title") or "", "body": b,
                         "cats": set(c),
                         "published_date": r.get("published_date")})
    return rows


def apply_rules(row: dict) -> tuple[set[str], set[str], list[str]]:
    pos, neg, fired = set(), set(), []
    for name, cats, p, n in RULES:
        if row["cats"] & cats:
            pos |= p
            neg |= n
            fired.append(name)
    neg -= pos          # a positive assertion wins over a negative one
    return pos, neg, fired


def audit(gold: dict[str, set[str]], corpus: list[dict]) -> None:
    by_url = {r["url"]: r for r in corpus}
    print(f"\n=== rule precision, measured on the {len(gold)} gold articles ===")
    print(f"  {'rule':34} {'gold n':>7} {'correct':>8} {'precision':>10} {'corpus':>8}")
    for name, cats, p, n in RULES:
        g = [u for u in gold if u in by_url and by_url[u]["cats"] & cats]
        pool = sum(1 for r in corpus if r["cats"] & cats)
        if not g:
            print(f"  {name:34} {0:>7} {'--':>8} {'--':>10} {pool:>8,}")
            continue
        lab = next(iter(p))
        ok = sum(1 for u in g if lab in gold[u])
        print(f"  {name:34} {len(g):>7} {ok:>8} {ok / len(g):>9.1%} {pool:>8,}")

    print(f"\n=== what the `other` rule asserts in full (it is exclusive) ===")
    g = [u for u in gold if u in by_url and by_url[u]["cats"] & SPORT]
    exact = sum(1 for u in g if gold[u] == {"other"})
    print(f"  gold articles matching sport|football: {len(g)}")
    print(f"  whose label set is EXACTLY {{other}}:     {exact}  ({exact / len(g):.1%})")
    print(f"  -> {sum(1 for r in corpus if r['cats'] & SPORT):,} corpus articles gain a "
          f"COMPLETE label set for free")

    print(f"\n=== per-label supervision the rules provide, corpus-wide ===")
    cnt = collections.Counter()
    for r in corpus:
        p, n, fired = apply_rules(r)
        for l in p:
            cnt[(l, "pos")] += 1
        for l in n:
            cnt[(l, "neg")] += 1
    gold_pos = {l: sum(1 for s in gold.values() if l in s) for l in humset.LABELS}
    print(f"  {'label':28} {'gold pos':>9} {'weak pos':>9} {'weak neg':>9}   verdict")
    for l in humset.LABELS:
        wp, wn = cnt[(l, "pos")], cnt[(l, "neg")]
        if wp == 0 and wn == 0:
            v = "NO RULE — gold only"
        elif wp == 0:
            v = "negatives only"
        else:
            v = f"{wp / max(gold_pos[l], 1):.0f}x more positives"
        print(f"  {l:28} {gold_pos[l]:9d} {wp:9,} {wn:9,}   {v}")


def build(corpus: list[dict]) -> list[dict]:
    out = []
    gold_urls = {json.loads(l)["url"] for l in open(GOLD)}
    for r in corpus:
        if r["url"] in gold_urls:
            continue                      # never leak the test set into training
        p, n, fired = apply_rules(r)
        if not p and not n:
            continue
        out.append({"url": r["url"], "site": r["site"], "title": r["title"],
                    "body": r["body"], "pos": sorted(p), "neg": sorted(n),
                    "rules": fired, "source": "weak:category"})
    return out


def fit_label(label: str, Xtr, pos_mask, Xte, C=0.5, maxf=300_000):
    """Train one binary classifier from rows that assert `label` either way."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=maxf, sublinear_tf=True, min_df=2)
    A = vec.fit_transform(Xtr)
    B = vec.transform(Xte)
    y = np.asarray(pos_mask, dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.full(B.shape[0], -9.9)
    clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000,
                    random_state=T.SEED)
    clf.fit(A, y)
    return clf.decision_function(B)


def evaluate(gold_rows, weak) -> None:
    """5-fold CV on gold, each fold trained with and without the weak pool."""
    X = [r["text"] for r in gold_rows]
    Y = [r["labels"] for r in gold_rows]
    sig = ["|".join(sorted(s)) for s in Y]
    c = collections.Counter(sig)
    sig = [s if c[s] >= 5 else "RARE" for s in sig]

    wtext = [w["title"] + "\n" + w["body"] for w in weak]
    print(f"\nweak pool: {len(weak):,} articles")

    scores = {m: [dict() for _ in X] for m in ("gold", "gold+weak")}
    skf = StratifiedKFold(5, shuffle=True, random_state=T.SEED)
    for k, (tr, te) in enumerate(skf.split(X, sig), 1):
        for label in humset.LABELS:
            gX = [X[i] for i in tr]
            gP = [label in Y[i] for i in tr]
            d = fit_label(label, gX, gP, [X[i] for i in te])
            for j, i in enumerate(te):
                scores["gold"][i][label] = float(d[j])

            use = [m for m, w in enumerate(weak)
                   if label in w["pos"] or label in w["neg"]]
            if use:
                wX = [wtext[m] for m in use]
                wP = [label in weak[m]["pos"] for m in use]
                d2 = fit_label(label, gX + wX, gP + wP, [X[i] for i in te])
            else:
                d2 = d
            for j, i in enumerate(te):
                scores["gold+weak"][i][label] = float(d2[j])
        print(f"  fold {k} done", flush=True)

    print(f"\n{'label':28} {'gold F1':>8} {'+weak F1':>9} {'delta':>7} {'weak rows':>10}")
    res = {}
    for mode in ("gold", "gold+weak"):
        P = []
        for i in range(len(X)):
            raw = {l for l in humset.LABELS if scores[mode][i][l] > 0}
            P.append(T.repair(scores[mode][i], raw))
        res[mode] = P
    for label in humset.LABELS:
        line = []
        for mode in ("gold", "gold+weak"):
            tp = sum(1 for y, p in zip(Y, res[mode]) if label in y and label in p)
            fp = sum(1 for y, p in zip(Y, res[mode]) if label not in y and label in p)
            fn = sum(1 for y, p in zip(Y, res[mode]) if label in y and label not in p)
            line.append(T.prf(tp, fp, fn)[2])
        nw = sum(1 for w in weak if label in w["pos"] or label in w["neg"])
        d = line[1] - line[0]
        flag = "" if nw else "   (no rule)"
        print(f"{label:28} {line[0]:8.3f} {line[1]:9.3f} {d:+7.3f} {nw:10,}{flag}")
        res[label] = line

    meas = [l for l in humset.LABELS
            if sum(1 for y in Y if l in y) >= T.MIN_SUPPORT]
    for mode in ("gold", "gold+weak"):
        f1 = []
        for label in meas:
            tp = sum(1 for y, p in zip(Y, res[mode]) if label in y and label in p)
            fp = sum(1 for y, p in zip(Y, res[mode]) if label not in y and label in p)
            fn = sum(1 for y, p in zip(Y, res[mode]) if label in y and label not in p)
            f1.append(T.prf(tp, fp, fn)[2])
        ex = sum(1 for y, p in zip(Y, res[mode]) if y == p) / len(Y)
        print(f"\n  {mode:10} macro-F1* {sum(f1) / len(f1):.3f}   exact {ex:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--cap", type=int, default=0,
                    help="cap the weak pool (0 = no cap)")
    args = ap.parse_args()
    if not (args.audit or args.build or args.evaluate):
        ap.error("pick --audit / --build / --evaluate")

    print("loading corpus ...", flush=True)
    corpus = load_corpus()
    gold = {json.loads(l)["url"]: set(json.loads(l)["labels"]) for l in open(GOLD)}
    print(f"{len(corpus):,} corpus articles, {len(gold)} gold")

    if args.audit:
        audit(gold, corpus)
    if args.build or args.evaluate:
        weak = build(corpus)
        if args.cap:
            import random
            weak = random.Random(T.SEED).sample(weak, min(args.cap, len(weak)))
        if args.build:
            with open(OUT, "w") as fh:
                for w in weak:
                    fh.write(json.dumps(w, ensure_ascii=False) + "\n")
            print(f"\nwrote {OUT}: {len(weak):,} weakly-labelled articles")
        if args.evaluate:
            text = {json.loads(l)["url"]: json.loads(l) for l in open(TASK)}
            gr = [{"url": u, "labels": s,
                   "text": (text[u].get("title") or "") + "\n" + (text[u].get("body") or "")}
                  for u, s in gold.items()]
            evaluate(gr, weak)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
