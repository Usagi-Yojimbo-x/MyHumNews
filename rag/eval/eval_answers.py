#!/usr/bin/env python3
"""Does the generated English answer hold up? Deliverable 3, measured.

    export GEMINI_API_KEY=...
    .venv/bin/python rag/eval/eval_answers.py            # all 49 questions
    .venv/bin/python rag/eval/eval_answers.py --only negative --k 8

Retrieval uses `ui.serve.Index` — the SAME code path the demo serves from, with
the same shipped configuration (whole-article BGE-M3 plus the places gazetteer
at alpha 0.10). Reusing the serving class rather than reimplementing retrieval
is deliberate: an evaluation that scores a different retriever than the one
users hit measures nothing about the system.

## Three things here need no human judgement

1. REFUSAL ON NEGATIVES. The retrieval evaluation proved a cosine threshold
   cannot reject out-of-scope questions -- known-item and negative top-1
   similarity distributions overlap completely, and "When does the next SpaceX
   Starship launch?" scored 0.647, above 24 of the 29 real questions. The fix
   had to be the generator judging support. These 5 questions test whether it
   does.
2. FALSE REFUSAL ON KNOWN-ITEMS. The same mechanism can fail the other way, by
   refusing answerable questions. Both rates matter; one alone is gameable by
   a generator that always refuses, or never does.
3. CITATION VALIDITY. Passages are numbered and the model must cite numbers,
   so a citation outside 1..n is a mechanically detectable hallucination. This
   is why it cites indices and not URLs, which can be fabricated plausibly.

Plus one automatic faithfulness PROXY: for known-item questions the gold
article URL is known, so "did it cite the gold article, given that retrieval
returned it" is computable. A real faithfulness judgement -- is each claim
actually supported by the cited passage -- still needs a human reading Burmese,
and this does not replace it.

## Caching

Every answer is appended to the cache JSONL and reused on the next run, keyed
by question id and k. The free tier is rate-limited, so a run that dies
halfway must not throw away the calls already paid for.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.generate import DEFAULT_MODEL, answer  # noqa: E402
from ui.serve import ALPHA, Index  # noqa: E402

HERE = Path(__file__).resolve().parent


def load_cache(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.open():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[d["key"]] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8,
                    help="passages passed to the generator")
    ap.add_argument("--only", choices=["known_item", "topical", "negative"],
                    help="one question type only")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="seconds between calls. The flash-lite models answer "
                         "in ~1.3s, which alone would fire ~46 req/min and "
                         "trip the free tier's limit; the bigger models take "
                         "~87s and need no spacing at all. 2.0 suits the fast "
                         "ones, and 429s are retried either way.")
    ap.add_argument("--cache", type=Path, default=HERE / "answers_cache.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "ANSWER_RESULTS.json")
    args = ap.parse_args()

    qs = [json.loads(l) for l in (HERE / "questions.jsonl").open()]
    if args.only:
        qs = [q for q in qs if q["type"] == args.only]
    print(f"{len(qs)} questions, k={args.k}, model={args.model}")

    idx = Index(str(ROOT / "rag/index"), str(ROOT / "rag/gazetteer/places.json"),
                str(ROOT / "data-vm/*/parsed/articles.jsonl"),
                snippet_chars=1800)

    cache = load_cache(args.cache)
    print(f"cache: {len(cache)} answers already on disk\n")
    rows, calls = [], 0

    for q in qs:
        qid = q.get("qid") or q.get("id") or q["question"][:40]
        key = f"{qid}|k{args.k}|{args.model}"
        hit = idx.search(q["question"], k=args.k, alpha=ALPHA)
        passages = [{"title": r["title"], "text": r["snippet"], "url": r["url"],
                     "outlet": r["outlet"], "published": r["published"]}
                    for r in hit["results"]]
        retrieved = [p["url"] for p in passages]

        if key in cache:
            a = cache[key]["answer"]
        else:
            if calls and args.sleep:
                time.sleep(args.sleep)
            try:
                a = answer(q["question"], passages, model=args.model)
            except RuntimeError as e:
                print(f"  {qid}  CALL FAILED: {e}")
                continue
            calls += 1
            with args.cache.open("a") as fh:
                fh.write(json.dumps({"key": key, "qid": qid,
                                     "question": q["question"],
                                     "type": q["type"], "answer": a,
                                     "retrieved": retrieved},
                                    ensure_ascii=False) + "\n")

        # questions.jsonl carries `target_urls`, a LIST -- a known-item
        # question can have more than one article that answers it.
        gold = set(q.get("target_urls") or [])
        cited = set(u for u in (a.get("cited_urls") or []) if u)
        rows.append({
            "qid": qid, "type": q["type"], "question": q["question"],
            "supported": bool(a.get("supported")),
            "n_citations": len(a.get("citations") or []),
            "invalid_citations": len(a.get("invalid_citations") or []),
            "n_gold": len(gold),
            "gold_retrieved": bool(gold & set(retrieved)),
            "gold_cited": bool(gold & cited),
            "answer": a.get("answer", ""),
            "unsupported_reason": a.get("unsupported_reason", ""),
        })
        flag = "OK " if a.get("supported") else "REF"
        print(f"  {qid:<6} {q['type']:<11} {flag} cites={len(a.get('citations') or [])} "
              f"{(a.get('answer') or a.get('unsupported_reason') or '')[:72]}")

    if not rows:
        sys.exit("\nno answers produced; is GEMINI_API_KEY set?")

    def grp(t):
        return [r for r in rows if r["type"] == t]

    neg, ki, top = grp("negative"), grp("known_item"), grp("topical")
    summary = {"model": args.model, "k": args.k, "alpha": ALPHA,
               "n_questions": len(rows), "new_calls": calls}

    print("\n" + "=" * 70)
    if neg:
        refused = sum(1 for r in neg if not r["supported"])
        summary["negative_refusal_rate"] = round(refused / len(neg), 3)
        print(f"  refusal on NEGATIVE questions      {refused}/{len(neg)}"
              f"   = {refused / len(neg):.0%}   (want high)")
    if ki:
        # Refusal must be judged against what retrieval ACTUALLY handed the
        # generator. A question whose gold article was never retrieved is a
        # retrieval miss, and refusing it is correct behaviour -- charging
        # generation for it measures the retriever instead.
        got = [r for r in ki if r["gold_retrieved"]]
        lost = [r for r in ki if not r["gold_retrieved"]]
        raw_ref = sum(1 for r in ki if not r["supported"])
        summary["known_item_refusal_rate_raw"] = round(raw_ref / len(ki), 3)
        print(f"  refusal on KNOWN-ITEM, unconditional {raw_ref}/{len(ki)}"
              f"   = {raw_ref / len(ki):.0%}   (confounded by retrieval)")
        if got:
            fr = sum(1 for r in got if not r["supported"])
            summary["false_refusal_given_gold_retrieved"] = round(fr / len(got), 3)
            print(f"  FALSE refusal, gold was retrieved    {fr}/{len(got)}"
                  f"   = {fr / len(got):.0%}   (want low -- the real metric)")
            cited = sum(1 for r in got if r["gold_cited"])
            summary["gold_cited_given_retrieved"] = round(cited / len(got), 3)
            print(f"  cited the gold article, when retrieved {cited}/{len(got)}"
                  f" = {cited / len(got):.0%}")
        if lost:
            answered = sum(1 for r in lost if r["supported"])
            summary["answered_without_gold"] = answered
            summary["n_gold_not_retrieved"] = len(lost)
            print(f"  answered anyway, gold NOT retrieved  {answered}/{len(lost)}"
                  f"   <- NEEDS HUMAN CHECK")
            print("     Either a different article genuinely answers it (the\n"
                  "     retrieval write-up found 4 such cases, since a 38k news\n"
                  "     corpus covers each event many times), or the generator\n"
                  "     stretched. Only a human reading Burmese can tell which.")
    if top:
        ref = sum(1 for r in top if not r["supported"])
        summary["topical_refusal_rate"] = round(ref / len(top), 3)
        print(f"  refusal on TOPICAL questions       {ref}/{len(top)}"
              f"   = {ref / len(top):.0%}   (no gold; informational)")

    bad = sum(r["invalid_citations"] for r in rows)
    withbad = sum(1 for r in rows if r["invalid_citations"])
    summary["invalid_citations_total"] = bad
    summary["answers_with_invalid_citations"] = withbad
    print(f"  hallucinated citation indices      {bad} across {withbad} answers")
    print("=" * 70)
    print("  Faithfulness proper -- is every claim supported by the cited\n"
          "  passage -- still needs a human reading Burmese. These numbers do\n"
          "  not substitute for it.")

    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
