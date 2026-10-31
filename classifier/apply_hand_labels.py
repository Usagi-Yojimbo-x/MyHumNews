#!/usr/bin/env python3
"""
Fold the hand labels into topics.jsonl, replacing the model's guess.

    python -m classifier.apply_hand_labels

A hand label is the best evidence in the project: a human read the article and
decided. Leaving topics.jsonl carrying a prediction for an article someone has
already adjudicated throws that away -- and for the ones where the model was
wrong, it keeps a known-wrong value in the deliverable.

Three outcomes, kept distinguishable so downstream can filter on any of them:

  human gave a label   -> topic = the human's label,  source = "hand"
  human pressed 's'    -> topic = null,               source = "hand"
                          reason = "not_an_article"
                          (a programme announcement has no topic; a prediction
                          for it is noise, so null is more honest than a guess)
  human pressed 'a'    -> topic = the model's label,  source = "predicted"
                          reason = "ambiguous"
                          (the taxonomy could not decide it, so the prediction
                          stands but is flagged as unreliable)

WHAT THIS DOES NOT DO: add the hand labels to TRAINING. That would be a
mistake, and not a small one. These 200 articles are the only unbiased
measurement of RFA that exists. Train on them and they stop being a test set --
the model would score well on them because it had seen them, and there would be
no way left to tell whether it had actually improved. If you want hand labels
for training, draw a SEPARATE sample; never reuse the evaluation set.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=os.path.join(HERE, "data", "labels.jsonl"))
    p.add_argument("--topics", default=os.path.join(HERE, "data", "topics.jsonl"))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    for f in (a.labels, a.topics):
        if not os.path.exists(f):
            print(f"missing {f}", file=sys.stderr)
            return 1

    hand = {}
    with open(a.labels, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                hand[r["url"]] = r

    rows = []
    with open(a.topics, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    stats = collections.Counter()
    changed = collections.Counter()

    for r in rows:
        h = hand.get(r["url"])
        if h is None:
            stats["untouched"] += 1
            continue
        before = r.get("topic")
        if h["label"]:
            r["topic"] = h["label"]
            r["source"] = "hand"
            r.pop("reason", None)
            stats["hand_label"] += 1
            if before != h["label"]:
                changed[f"{before} -> {h['label']}"] += 1
                stats["corrected"] += 1
        elif h.get("reason") == "ambiguous":
            r["source"] = "predicted"
            r["reason"] = "ambiguous"
            stats["flagged_ambiguous"] += 1
        else:
            r["topic"] = None
            r["source"] = "hand"
            r["reason"] = "not_an_article"
            stats["cleared_non_article"] += 1

    print("=== applying hand labels to topics.jsonl ===")
    for k in ("hand_label", "corrected", "cleared_non_article",
              "flagged_ambiguous", "untouched"):
        if stats[k]:
            print(f"  {k:22}{stats[k]:>8}")
    if stats["hand_label"]:
        print(f"\n  model was right on {stats['hand_label'] - stats['corrected']}"
              f"/{stats['hand_label']} of the hand-labelled articles")

    if changed:
        print("\n  corrections applied (model -> human):")
        for k, v in changed.most_common(12):
            print(f"    {v:>4}  {k}")

    if a.dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    backup = a.topics + f".bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(a.topics, backup)
    tmp = a.topics + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, a.topics)

    src = collections.Counter(r["source"] for r in rows)
    notopic = sum(1 for r in rows if r.get("topic") is None)
    print(f"\n  backed up to {os.path.basename(backup)}")
    print(f"  wrote {a.topics}")
    print(f"    rows {len(rows)}   by source {dict(src)}")
    print(f"    rows with no topic (non-articles): {notopic}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
