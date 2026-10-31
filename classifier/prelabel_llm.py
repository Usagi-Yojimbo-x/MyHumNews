#!/usr/bin/env python3
"""
LLM pre-labelling for the 7 HumSet labels, at corpus scale.

    export ANTHROPIC_API_KEY=sk-ant-...
    python -m classifier.prelabel_llm --estimate              # cost, no API calls
    python -m classifier.prelabel_llm --uniform 8000          # training pool
    python -m classifier.prelabel_llm --target humanitarian_access --n 600
    python -m classifier.prelabel_llm --validate              # score vs the 383 gold

This is the "Label (LLM pre-labels)" step of the proposal pipeline. Its output
is TRAINING data for `train_humset.py --train`. It must never be merged into
`humset_gold.jsonl`: the proposal's metric is macro-F1 on a HAND-CHECKED test
set, and gold contaminated with model labels measures the labeller, not the
classifier.

Uses stdlib urllib, not the `anthropic` SDK, to avoid adding a dependency to a
project that currently has none beyond scikit-learn.

WHY NOT SIMPLY LABEL ALL 38,152
-------------------------------
Two reasons, one of cost and one of measurement.

Cost: ~2,040 chars/article mean. Burmese tokenizes badly -- roughly 1.3-1.6
tokens per character against ~0.25 for English -- so an article is ~2,700-3,300
input tokens, not the ~500 an English article of that length would be. The
whole corpus is therefore on the order of 110M input tokens. `--estimate`
computes this from the actual corpus rather than from this comment.

Measurement: a uniform sample is the WRONG shape for the two labels that are
currently unmeasurable. `humanitarian_access` is 4.2% of articles and
`information_communication` 1.6%, so a uniform 8,000 would be expected to add
~340 and ~130 of them -- fine -- but the learning curve (README Phase 7) shows
F1 reaches only ~0.5 at 30 positives and the real need is per-label coverage,
not raw volume. `--target` goes after a specific label directly.

HOW --target RANKS
------------------
It trains the one-vs-rest scorer on all 383 gold articles for that label alone,
scores the corpus, and sends the top `--n` by decision margin for labelling.
This is standard uncertainty-free active learning (most-confident-positive
sampling), chosen over keyword seeds because inventing Burmese keyword lists
would bake in the author's guesses about the language.

CAVEAT, stated because it matters: for `humanitarian_access` and
`information_communication` the gold-trained classifier scores F1 0.000, so its
RANKING is unverified. It may still be far better than the 4.2%/1.6% base rate
-- ranking survives where thresholding fails, which is exactly what Phase 4
found for margins -- but that is an assumption until the returned labels show
the hit rate. The script reports that hit rate, so the assumption gets tested
on the first run. If it comes back near the base rate, the ranking is worthless
and keyword seeding is the fallback.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from classifier import humset

API = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-5-5"
CORPUS = Path("data-vm")
SEED = 20261015
MAX_BODY = 6000          # chars; p90 of the gold set is 3,629
TOK_PER_CHAR = 1.45      # Burmese, measured range 1.3-1.6

# Prices are per million tokens and WILL go stale. --estimate prints them so
# the number is auditable rather than implicit.
PRICE_IN, PRICE_OUT, PRICE_CACHE_READ = 0.10, 0.50, 0.01


def build_system() -> str:
    """The labelling instructions. Mirrors HUMSET_GUIDE.md deliberately -- the
    model and the human annotators must be held to the same definitions, or
    agreement between them measures the prompt rather than the task."""
    lines = [
        "You label Burmese-language news articles about Myanmar with "
        "humanitarian categories, for a humanitarian news monitoring corpus.",
        "",
        "Apply ALL labels that fit. Most articles take 1-2.",
        "",
    ]
    for lab in humset.LABELS:
        lines.append(f"{lab}: {humset.DEFINITION[lab]}")
    lines += [
        "",
        "HARD RULES:",
        f"- `{humset.EXCLUSIVE}` is EXCLUSIVE. If it applies, it is the only label.",
        "- `other` means NO humanitarian relevance AT ALL, not 'no emergency'.",
        "  Myanmar politics, economy, sanctions, diplomacy, detention, elections",
        "  and trade are `context`, NOT `other`. The test: would someone",
        "  monitoring Myanmar's humanitarian situation want to know this?",
        "  Sport, entertainment, and foreign news with no Myanmar angle are `other`.",
        "- `casualties` requires `shock_event` as well: deaths and injuries come",
        "  from an event. Never return `casualties` alone.",
        "- Never return an empty list.",
        "",
        "Reply with ONLY a JSON object, no prose:",
        '{"labels": ["..."], "confidence": "high"|"medium"|"low"}',
    ]
    return "\n".join(lines)


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
                continue           # the <200-char bodies are the known bad layer
            seen.add(u)
            rows.append({"url": u, "site": r.get("outlet"), "title": r.get("title") or "",
                         "body": b, "published_date": r.get("published_date")})
    return rows


def est_tokens(r: dict) -> int:
    return int((len(r["title"]) + min(len(r["body"]), MAX_BODY)) * TOK_PER_CHAR) + 40


def estimate(rows: list[dict], n: int | None) -> None:
    # A RANDOM sample, not rows[:n]. The corpus is loaded in glob order, so the
    # first n articles are all one outlet -- BBC, whose bodies run ~30% longer
    # than the corpus mean -- and slicing overstated the cost by that much.
    sub = rows if n is None else random.Random(SEED).sample(rows, min(n, len(rows)))
    tin = sum(est_tokens(r) for r in sub)
    sys_tok = int(len(build_system()) * 0.3)
    tout = 30 * len(sub)
    cost_fresh = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    cached = (sys_tok * len(sub)) / 1e6 * PRICE_CACHE_READ
    print(f"\n  articles                {len(sub):,}")
    print(f"  mean body chars         {sum(len(r['body'][:MAX_BODY]) for r in sub) / len(sub):,.0f}")
    print(f"  est. input tokens       {tin / 1e6:.1f}M   (at {TOK_PER_CHAR} tok/char, Burmese)")
    print(f"  est. output tokens      {tout / 1e6:.2f}M")
    print(f"  system prompt           {sys_tok:,} tok, cached across calls")
    print(f"\n  at ${PRICE_IN}/${PRICE_OUT} per Mtok in/out, ${PRICE_CACHE_READ} cache read:")
    print(f"    content              ${cost_fresh:,.2f}")
    print(f"    cached system        ${cached:,.2f}")
    print(f"    TOTAL                ${cost_fresh + cached:,.2f}")
    print("\n  (prices hardcoded in this file and will go stale -- check them)")


def call(r: dict, system: str, key: str, retries: int = 5) -> dict | None:
    body = json.dumps({
        "model": MODEL,
        "max_tokens": 100,
        "system": [{"type": "text", "text": system,
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content":
                      f"TITLE: {r['title']}\n\nBODY:\n{r['body'][:MAX_BODY]}"}],
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01",
        "content-type": "application/json"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                d = json.loads(resp.read())
            txt = d["content"][0]["text"].strip()
            txt = txt[txt.find("{"): txt.rfind("}") + 1]
            parsed = json.loads(txt)
            labels = [l for l in parsed.get("labels", []) if l in humset.LABELS]
            if not labels:
                return None
            if humset.EXCLUSIVE in labels:
                labels = [humset.EXCLUSIVE]
            if "casualties" in labels and "shock_event" not in labels:
                labels.append("shock_event")        # the guide's hard rule
            return {"url": r["url"], "site": r["site"],
                    "published_date": r["published_date"],
                    "labels": sorted(set(labels)),
                    "confidence": parsed.get("confidence"),
                    "annotator": f"llm:{MODEL}", "pass": "llm"}
        except (urllib.error.HTTPError, urllib.error.URLError, OSError,
                KeyError, ValueError, json.JSONDecodeError) as e:
            code = getattr(e, "code", None)
            if code in (400, 401, 403):
                sys.exit(f"fatal API error {code}: {e}")
            time.sleep(min(2 ** a, 30))
    return None


def run(rows: list[dict], out: Path, workers: int) -> None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY is not set. "
                 "Run with --estimate to size the job without it.")
    done = set()
    if out.exists():
        done = {json.loads(l)["url"] for l in open(out) if l.strip()}
        print(f"resuming: {len(done):,} already labelled in {out}")
    todo = [r for r in rows if r["url"] not in done]
    print(f"labelling {len(todo):,} articles with {workers} workers -> {out}")
    system = build_system()
    lock, n_ok, n_fail = threading.Lock(), 0, 0
    t0 = time.time()
    with open(out, "a") as fh, ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(lambda r: call(r, system, key), todo), 1):
            with lock:
                if res is None:
                    n_fail += 1
                else:
                    fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                    n_ok += 1
                if i % 100 == 0:
                    fh.flush()
                    rate = i / (time.time() - t0)
                    print(f"  {i:,}/{len(todo):,}  ok={n_ok:,} fail={n_fail:,}  "
                          f"{rate:.1f}/s  eta {(len(todo) - i) / max(rate, .01) / 60:.0f}m",
                          flush=True)
    print(f"\ndone: {n_ok:,} labelled, {n_fail:,} failed -> {out}")


def rank_for(label: str, rows: list[dict], n: int) -> list[dict]:
    """Most-confident-positive sampling for one label."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from classifier import train_humset as T

    X, Y, _ = T.build(Path("classifier/data/humset_gold.jsonl"),
                      Path("classifier/data/humset_task.jsonl"))
    pos = sum(1 for y in Y if label in y)
    print(f"  training ranker for `{label}` on {len(X)} gold articles "
          f"({pos} positives)")
    if pos < T.MIN_SUPPORT:
        print(f"  WARNING: {pos} positives is below MIN_SUPPORT={T.MIN_SUPPORT}. "
              f"This label's classifier scores F1 0.000, so the RANKING below is "
              f"UNVERIFIED. The hit rate reported after labelling tests it.")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=300_000, sublinear_tf=True, min_df=2)
    A = vec.fit_transform(X)
    clf = LinearSVC(C=0.5, class_weight="balanced", max_iter=5000,
                    random_state=SEED)
    clf.fit(A, np.array([1 if label in y else 0 for y in Y]))

    gold_urls = {json.loads(l)["url"]
                 for l in open("classifier/data/humset_gold.jsonl")}
    cand = [r for r in rows if r["url"] not in gold_urls]
    B = vec.transform([r["title"] + "\n" + r["body"] for r in cand])
    s = clf.decision_function(B)
    order = s.argsort()[::-1][:n]
    print(f"  base rate in gold: {pos / len(X):.1%};  "
          f"corpus scored above 0: {(s > 0).sum():,} of {len(cand):,} ({(s > 0).mean():.1%})")
    print(f"  taking top {n:,} by margin (scores {s[order[0]]:.2f} .. {s[order[-1]]:.2f})")
    return [cand[i] for i in order]


def validate(pred_path: Path) -> None:
    """Score the LLM against the 383 hand labels -- the labeller's own accuracy."""
    gold = {json.loads(l)["url"]: set(json.loads(l)["labels"])
            for l in open("classifier/data/humset_gold.jsonl")}
    pred = {}
    for l in open(pred_path):
        r = json.loads(l)
        if r["url"] in gold:
            pred[r["url"]] = set(r["labels"])
    if not pred:
        sys.exit(f"{pred_path} contains none of the 383 gold urls. "
                 f"Label them first: --validate-set")
    print(f"\nLLM vs hand labels on {len(pred)} of {len(gold)} gold articles")
    from classifier import train_humset as T
    for lab in humset.LABELS:
        tp = sum(1 for u in pred if lab in gold[u] and lab in pred[u])
        fp = sum(1 for u in pred if lab not in gold[u] and lab in pred[u])
        fn = sum(1 for u in pred if lab in gold[u] and lab not in pred[u])
        p, r_, f = T.prf(tp, fp, fn)
        print(f"  {lab:28} P {p:.3f}  R {r_:.3f}  F1 {f:.3f}  supp {tp + fn:4d}")
    ex = sum(1 for u in pred if gold[u] == pred[u])
    print(f"  exact match {ex / len(pred):.3f}")
    print("\n  This is the CEILING for anything trained on these pre-labels.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estimate", action="store_true", help="cost only, no API calls")
    ap.add_argument("--uniform", type=int, metavar="N",
                    help="label a uniform random sample of N articles")
    ap.add_argument("--target", choices=humset.LABELS,
                    help="label the N articles most likely to carry this label")
    ap.add_argument("--validate-set", action="store_true",
                    help="label the 383 gold articles, to measure the labeller")
    ap.add_argument("--validate", type=Path, metavar="PRED",
                    help="score an existing pre-label file against gold")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", type=Path, default=Path("classifier/data/prelabels.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.validate:
        validate(args.validate)
        return 0

    print("loading corpus from data-vm/ ...", flush=True)
    rows = load_corpus()
    print(f"{len(rows):,} articles with usable bodies "
          f"({collections.Counter(r['site'] for r in rows).most_common()})")

    if args.estimate:
        print("\n=== whole corpus ===")
        estimate(rows, None)
        for n in (2000, 8000):
            print(f"\n=== uniform {n:,} ===")
            estimate(rows, n)
        return 0

    if args.validate_set:
        gold_urls = {json.loads(l)["url"]
                     for l in open("classifier/data/humset_gold.jsonl")}
        sel = [r for r in rows if r["url"] in gold_urls]
        args.out = args.out.with_name("prelabels_goldset.jsonl")
        print(f"labelling the {len(sel)} gold articles to measure the labeller")
    elif args.target:
        sel = rank_for(args.target, rows, args.n)
        args.out = args.out.with_name(f"prelabels_{args.target}.jsonl")
    elif args.uniform:
        rng = random.Random(SEED)
        sel = rng.sample(rows, min(args.uniform, len(rows)))
    else:
        ap.error("pick one of --estimate / --uniform / --target / --validate-set / --validate")

    estimate(sel, None)
    run(sel, args.out, args.workers)

    if args.target:
        got = [json.loads(l) for l in open(args.out) if l.strip()]
        hit = sum(1 for r in got if args.target in r["labels"])
        print(f"\n  HIT RATE for `{args.target}`: {hit}/{len(got)} = {hit / len(got):.1%}")
        print(f"  gold base rate was "
              f"{sum(1 for l in open('classifier/data/humset_gold.jsonl') if args.target in json.loads(l)['labels']) / 383:.1%}")
        print("  If these are close, the ranking added nothing and keyword "
              "seeding is the fallback.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
