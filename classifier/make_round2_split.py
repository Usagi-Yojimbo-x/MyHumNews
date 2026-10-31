"""Split the round-2 thin-label set into equal train/test halves.

Round 2 is an ENRICHED sample: a ranker chose 360 articles likely to carry
`humanitarian_access` or `information_communication`, the two labels the 383
gold articles have almost none of (16 and 6). It is therefore NOT a random
sample of the corpus and must never be merged into `humset_gold.jsonl`, which
remains the only population-shaped ground truth.

Both thin labels came back at a 50%+ hit rate, so splitting costs nothing:
each half still clears the ~45-positive point where the Phase 7 learning curve
saturates, and the test half buys the first measurable estimate for labels that
have been stuck at F1 = 0.000 in both arms.

Stratified on (targeted label x whether that label was actually applied), with
outlets interleaved inside each cell, so the halves match on hit rate, outlet
mix and label counts rather than only on size.

Numbers from this set are CONDITIONAL on the enriched stratum. The bias runs
both ways -- recall optimistic because the ranker surfaced easy positives,
precision pessimistic because it also surfaced hard negatives -- so the net
direction is indeterminate. Report as "on a targeted sample", never as a
corpus-wide rate.
"""
import argparse
import collections
import json
import random
from pathlib import Path

import humset

SEED = 20261016
KEEP = ("url", "site", "annotator", "labels", "published_date", "labelled_at",
        "reason", "i")


def load(labels: Path, task: Path):
    task_by_url = {}
    for line in open(task):
        r = json.loads(line)
        task_by_url[r["url"]] = r
    rows = []
    for line in open(labels):
        r = json.loads(line)
        if r.get("annotator") != "HsuLinHtet":
            continue                      # stray test clicks from other users
        t = task_by_url[r["url"]]
        out = {k: r.get(k) for k in KEEP}
        out["site"] = humset.canon_site(out["site"])   # dvb_burmese -> dvb
        out["targeted_for"] = t["_targeted_for"]
        out["margin"] = t["_margin"]
        out["n_chars"] = t["n_chars"]
        out["labels"] = sorted(r["labels"])
        rows.append(out)                  # NOTE: no body -- copyrighted
    return rows, task_by_url


def assign(rows, rng):
    """Interleave outlets inside each (target, hit) cell, then alternate."""
    cells = collections.defaultdict(list)
    for r in rows:
        hit = r["targeted_for"] in r["labels"]
        cells[(r["targeted_for"], hit)].append(r)

    for key in sorted(cells):
        by_outlet = collections.defaultdict(list)
        for r in cells[key]:
            by_outlet[r["site"]].append(r)
        for v in by_outlet.values():
            rng.shuffle(v)
        order, pools = [], [by_outlet[s] for s in sorted(by_outlet)]
        while any(pools):
            for pool in pools:
                if pool:
                    order.append(pool.pop())
        flip = rng.random() < 0.5
        for n, r in enumerate(order):
            r["split"] = "train" if (n % 2 == 0) != flip else "test"
    return rows


def imbalance(rows):
    """Total train/test count gap over every label and every outlet.

    The two targeted strata are halved exactly by construction in any
    assignment, so this only measures the drift that stratification does not
    control: the five incidental labels and the outlet mix.
    """
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    gap = 0
    for l in humset.LABELS:
        gap += abs(sum(1 for r in tr if l in r["labels"])
                   - sum(1 for r in te if l in r["labels"]))
    for s in {r["site"] for r in rows}:
        gap += abs(sum(1 for r in tr if r["site"] == s)
                   - sum(1 for r in te if r["site"] == s))
    return gap


def split(rows, trials=2000):
    """Pick the most balanced of `trials` stratified assignments.

    Every candidate already splits the targeted strata exactly in half, so this
    search cannot trade away the property the split exists for -- it only picks
    the candidate whose incidental labels and outlets line up best.
    """
    best, best_gap, best_i = None, None, None
    for i in range(trials):
        cand = assign(rows, random.Random(SEED + i))
        gap = imbalance(cand)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_i = i
            best = {r["url"]: r["split"] for r in cand}
    for r in rows:
        r["split"] = best[r["url"]]
    print(f"balance search: {trials} stratified assignments, "
          f"best is seed offset {best_i} with total gap {best_gap} "
          f"(worst tried {max(imbalance(assign(rows, random.Random(SEED + i))) for i in range(min(trials, 50)))})")
    for r in rows:
        r["split"] = best[r["url"]]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--task", type=Path,
                    default=Path("classifier/data/humset_thin_task.jsonl"))
    ap.add_argument("--gold", type=Path,
                    default=Path("classifier/data/humset_gold.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("classifier/data/humset_round2.jsonl"))
    ap.add_argument("--trials", type=int, default=2000,
                    help="stratified assignments to try; the most "
                         "balanced one wins")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows, task = load(args.labels, args.task)
    gold = {json.loads(l)["url"] for l in open(args.gold)}

    # Guard rails. Any of these failing means the split would be wrong.
    assert len(rows) == len(task) == 360, (len(rows), len(task))
    assert len({r["url"] for r in rows}) == len(rows), "duplicate urls"
    leak = {r["url"] for r in rows} & gold
    assert not leak, f"{len(leak)} round-2 urls are already in gold"
    for r in rows:
        assert r["labels"], f"empty label set: {r['url']}"
        assert not (humset.EXCLUSIVE in r["labels"] and len(r["labels"]) > 1)
        for l in r["labels"]:
            assert l in humset.LABELS, l

    rows = split(rows, args.trials)
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    print(f"round 2: {len(rows)} articles -> train {len(tr)}  test {len(te)}")
    print(f"gold untouched: {len(gold)} articles, overlap 0\n")

    print(f"  {'label':28} {'train':>6} {'test':>6} {'total':>6}")
    for l in humset.LABELS:
        a = sum(1 for r in tr if l in r["labels"])
        b = sum(1 for r in te if l in r["labels"])
        g = sum(1 for line in open(args.gold)
                if l in json.loads(line)["labels"])
        print(f"  {l:28} {a:>6} {b:>6} {a + b:>6}   (gold {g})")

    print(f"\n  {'stratum':34} {'train':>14} {'test':>14}")
    for t in ("humanitarian_access", "information_communication"):
        a = [r for r in tr if r["targeted_for"] == t]
        b = [r for r in te if r["targeted_for"] == t]
        ha = sum(1 for r in a if t in r["labels"])
        hb = sum(1 for r in b if t in r["labels"])
        print(f"  {t:34} {ha:>4}/{len(a):<3} {ha / len(a):>6.1%} "
              f"{hb:>4}/{len(b):<3} {hb / len(b):>6.1%}")

    print(f"\n  {'outlet':20} {'train':>6} {'test':>6}")
    for s in sorted({r["site"] for r in rows}):
        print(f"  {s:20} {sum(1 for r in tr if r['site'] == s):>6} "
              f"{sum(1 for r in te if r['site'] == s):>6}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}  ({len(rows)} rows, no article bodies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
