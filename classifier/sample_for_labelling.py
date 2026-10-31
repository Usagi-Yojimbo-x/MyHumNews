#!/usr/bin/env python3
"""
Draw a blind sample of RFA articles for hand-labelling.

    python -m classifier.sample_for_labelling --n 200

WHY SIMPLE RANDOM, not stratified by predicted topic:
the point of this exercise is an unbiased estimate of how accurate the model is
on RFA, and only a random sample gives one. Stratifying by the model's own
prediction would let you measure per-class precision, but it cannot measure
recall at all -- you would never sample the articles the model wrongly moved
OUT of a class, because you are sampling by where it put them. It also makes
the headline accuracy a weighted reconstruction rather than a direct count,
which is one more thing to get wrong.

The cost is thin coverage of rare classes. At RFA's predicted distribution a
random 200 yields roughly: conflict 86, politics 51, crime 17, environment 15,
economy 14, international 10, and ~3 or fewer each of culture/health/education/
sport. That measures the top six -- 96% of RFA's articles -- properly, and
leaves the tail unmeasured. If per-class numbers for the tail matter later, draw
a SEPARATE top-up sample and keep it separate; do not dilute this one.

TWO SOURCES, for two different questions:

  --source predicted  (default)  articles with NO outlet label, e.g. all of RFA.
      The only question available is "is the model right?", because there is
      nothing else to compare against.

  --source heldout               articles from test.jsonl, which DO have an outlet
      label and which the model never trained on. This permits the measurement
      that actually matters:

          model vs hand   the model's real accuracy, independent of the
                          outlet's own labels
          outlet vs hand  how good the OUTLET'S section names are as topic
                          labels -- the circularity test
          model vs outlet already known per outlet (dvb 0.875)

      If outlet-vs-hand agreement is high, the section names are a sound
      proxy for topic and the held-out accuracy means what it appears to mean.
      If it is low, then that accuracy is largely the model reproducing a noisy
      label source, and every figure resting on it has to be restated.

      Sampling from test.jsonl, not from all of DVB, is essential: the model
      trained on the other 80%, so a sample drawn from everything would measure
      memorisation.

WHY THE SAMPLE IS BLIND:
the task file deliberately does NOT contain the model's prediction. Showing it
would anchor the labeller toward agreeing, and an accuracy measured against a
prediction the labeller could see is not a measurement. score_labels.py joins
the labels back to predicted.jsonl afterwards, by url.

The sample is seeded, so it is reproducible and re-running will not silently
draw a different 200.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data-vm")
    p.add_argument("--site", default="rfa")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--source", choices=["predicted", "heldout"],
                   default="predicted",
                   help="predicted: unlabelled articles. heldout: test.jsonl, "
                        "which carries outlet labels the model never saw.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default=os.path.join(HERE, "data", "label_task.jsonl"))
    a = p.parse_args()

    if a.source == "heldout":
        test_path = os.path.join(HERE, "data", "test.jsonl")
        want = set()
        with open(test_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r["site"] == a.site:
                        want.add(r["url"])
        if not want:
            print(f"no held-out articles for site={a.site}", file=sys.stderr)
            return 1
    else:
        # Only articles that actually got a prediction -- that is the
        # population whose accuracy we are trying to estimate.
        pred_path = os.path.join(HERE, "data", "predicted.jsonl")
        want = set()
        with open(pred_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r["site"] == a.site:
                        want.add(r["url"])
        if not want:
            print(f"no predicted articles for site={a.site}", file=sys.stderr)
            return 1

    pool = []
    src = os.path.join(a.data, a.site, "parsed", "articles.jsonl")
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["url"] in want:
                pool.append(r)

    # Dedupe on url, keeping the first -- a url appearing twice must not get two
    # chances to be drawn.
    seen, uniq = set(), []
    for r in pool:
        if r["url"] not in seen:
            seen.add(r["url"])
            uniq.append(r)

    if len(uniq) < a.n:
        print(f"only {len(uniq)} available, sampling all of them", file=sys.stderr)
        a.n = len(uniq)

    sample = random.Random(a.seed).sample(uniq, a.n)
    # Present in corpus order, not draw order: a labeller working through a
    # chronological run holds context better than one jumping across two years.
    sample.sort(key=lambda r: r.get("published_date") or "")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(sample, 1):
            # `category`/`categories` are the OUTLET LABEL SOURCE and are
            # deliberately not written. For DVB they would hand the answer
            # straight to the labeller. `tags` is free-text keywords and is
            # safe, but is dropped too when a outlet label exists, since an
            # outlet's tags can echo its sections.
            task = {
                "i": i,
                "url": r["url"],
                "site": a.site,
                "title": r.get("title") or "",
                "body": r.get("body") or "",
                "published_date": r.get("published_date"),
                "tags": [] if a.source == "heldout" else (r.get("tags") or []),
                "author": r.get("author") or "",
                "n_chars": len(r.get("body") or ""),
            }
            assert "topic" not in task and "categories" not in task
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")

    short = sum(1 for r in sample if len(r.get("body") or "") < 200)
    kind = ("held-out (outlet-labelled, never trained on)"
            if a.source == "heldout" else "predicted (no outlet label)")
    print(f"population: {len(uniq)} {a.site} articles -- {kind}")
    print(f"drew       : {a.n} (seed {a.seed}, reproducible)")
    print(f"date range : {sample[0]['published_date'][:10]} .. "
          f"{sample[-1]['published_date'][:10]}")
    print(f"under 200 chars: {short}  (mark these 's' for skip if unlabelable)")
    print(f"\nwrote {a.out}")
    print(f"  blind: no predictions, no outlet labels, no category fields")
    print(f"\nnext:  python -m classifier.label_cli")
    return 0


if __name__ == "__main__":
    sys.exit(main())
