"""Draw a blind second-pass sample for inter-annotator agreement.

Excludes every gold article the second annotator has already seen, so the pass
is genuinely blind: the 85 they re-labelled in pass 2, and the 50 served in IAA
run 1. The task file carries titles and bodies only -- never pass-1 labels.

KNOWN LIMIT, structural and not fixable by sampling: the second annotator has
already seen all 85 articles pass 1 called `other`, because those were exactly
the pass-2 re-labelling set. Any unseen sample therefore contains zero pass-1
`other` articles, and this design can never measure agreement on that label.
The 0.812 exact match from pass 2 is the only `other` figure available, and it
is not comparable -- that subset was selected on pass 1's label.
"""
import argparse
import collections
import json
import random
from pathlib import Path

SEED = 20261016


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path,
                    default=Path("classifier/data/humset_gold.jsonl"))
    ap.add_argument("--bodies", type=Path,
                    default=Path("classifier/data/humset_task.jsonl"))
    ap.add_argument("--annotator", default="ZweHtetAung",
                    help="whose prior exposure to exclude")
    ap.add_argument("--seen", type=Path, nargs="*", default=[],
                    help="earlier task files this annotator was served")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gold = [json.loads(l) for l in open(args.gold)]
    bodies = {json.loads(l)["url"]: json.loads(l) for l in open(args.bodies)}

    seen = {r["url"] for r in gold if r.get("annotator") == args.annotator}
    print(f"gold {len(gold)}  already re-labelled by {args.annotator}: {len(seen)}")
    for p in args.seen:
        prev = {json.loads(l)["url"] for l in open(p)}
        seen |= prev
        print(f"  + {len(prev)} from {p.name}")

    pool = [r for r in gold if r["url"] not in seen]
    print(f"unseen pool: {len(pool)}")
    assert len(pool) >= args.n, "pool too small"
    missing = [r["url"] for r in pool if r["url"] not in bodies]
    assert not missing, f"{len(missing)} pool articles have no body available"

    # Stratify on label-set signature so the sample mirrors the pool's mix of
    # easy single-label and hard multi-label articles, not just its size.
    sig = collections.defaultdict(list)
    for r in pool:
        sig["|".join(sorted(r["labels"]))].append(r)

    # Largest-remainder allocation: every stratum gets its proportional share,
    # and the leftover seats go to the strata with the largest fractions. This
    # is fixed, so the label mix is right in every candidate below.
    quota = {k: len(v) * args.n / len(pool) for k, v in sig.items()}
    take = {k: int(q) for k, q in quota.items()}
    for k in sorted(sig, key=lambda k: (-(quota[k] - take[k]), k)):
        if sum(take.values()) >= args.n:
            break
        take[k] += 1

    labels = sorted({l for r in pool for l in r["labels"]})
    sites = sorted({r["site"] for r in pool})

    def deviation(sample):
        """How far the sample's label and outlet shares sit from the pool's.

        The allocation above already fixes the label mix, so in practice this
        is picking between candidates on outlet balance -- but scoring both
        keeps a candidate from buying outlet balance with a label skew.
        """
        d = 0.0
        for l in labels:
            d += abs(sum(1 for r in sample if l in r["labels"]) / len(sample)
                     - sum(1 for r in pool if l in r["labels"]) / len(pool))
        for st in sites:
            d += abs(sum(1 for r in sample if r["site"] == st) / len(sample)
                     - sum(1 for r in pool if r["site"] == st) / len(pool))
        return d

    best, best_d, best_i = None, None, None
    for i in range(args.trials):
        rng = random.Random(SEED + i)
        cand = []
        for k in sorted(sig):
            v = list(sig[k])
            rng.shuffle(v)
            cand += v[:take[k]]
        d = deviation(cand)
        if best_d is None or d < best_d:
            best, best_d, best_i = cand, d, i
    print(f"\nbalance search: {args.trials} stratified draws, best is offset "
          f"{best_i} with total share deviation {best_d:.3f}")
    picked = best
    random.Random(SEED).shuffle(picked)
    assert len(picked) == args.n, len(picked)
    assert not ({r["url"] for r in picked} & seen), "leak: annotator has seen one"

    print(f"\n  {'label':28} {'pool':>6} {'sample':>7} {'pool %':>8} {'samp %':>8}")
    for l in sorted({l for r in pool for l in r["labels"]}):
        a = sum(1 for r in pool if l in r["labels"])
        b = sum(1 for r in picked if l in r["labels"])
        print(f"  {l:28} {a:>6} {b:>7} {a / len(pool):>7.1%} {b / len(picked):>7.1%}")
    print(f"\n  outlets  pool {dict(collections.Counter(r['site'] for r in pool))}")
    print(f"           samp {dict(collections.Counter(r['site'] for r in picked))}")
    print(f"  cardinality samp "
          f"{dict(sorted(collections.Counter(len(r['labels']) for r in picked).items()))}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    with open(args.out, "w") as fh:
        for n, r in enumerate(picked, 1):
            b = bodies[r["url"]]
            fh.write(json.dumps({      # NO labels -- the pass must be blind
                "i": n, "url": r["url"], "site": r["site"],
                "title": b["title"], "body": b["body"],
                "published_date": r.get("published_date"),
                "n_chars": len(b["body"]),
            }, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out} ({args.n} articles, no labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
