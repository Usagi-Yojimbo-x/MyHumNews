#!/usr/bin/env python3
"""
Reopen a subset of labels for a second pass.

    python -m classifier.relabel --topic international --skipped-substantive

Removes the matching rows from labels.jsonl (backing it up first), so
label_web / label_cli serve those articles again. Everything else is untouched.

Why this exists rather than hand-editing the file: the first pass surfaced two
problems that need a targeted redo, not a restart.

  1. 27 articles labelled `international` that are Myanmar stories with a
     foreign actor. The outlet label `international` means foreign news with
     NO Myanmar angle -- only 1.6% of 8,674 such titles mention Myanmar,
     while Myanmar+foreign-actor stories go to politics (28%), economy (21%),
     conflict (19%) and crime (15%). The convention was in the guide but buried.

  2. 42 substantive articles marked `s` with `a` never used. `s` means "not an
     article"; these are real news, mostly on the conflict/politics boundary
     the model is worst at. Scoring without them measures the easy subset and
     flatters the model.

Flags:
  --topic X                 reopen everything labelled X
  --skipped-substantive     reopen skips that are NOT programme boilerplate
  --dry-run                 show what would be reopened, change nothing
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BOILERPLATE = "RFA ရေဒီယိုနဲ့ရုပ်သံ"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=os.path.join(HERE, "data", "labels.jsonl"))
    p.add_argument("--task", default=os.path.join(HERE, "data", "label_task.jsonl"))
    p.add_argument("--topic", action="append", default=[],
                   help="reopen all rows with this label (repeatable)")
    p.add_argument("--skipped-substantive", action="store_true",
                   help="reopen skips whose body is not programme boilerplate")
    p.add_argument("--min-chars", type=int, default=400,
                   help="a skip under this length is treated as boilerplate")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if not os.path.exists(a.labels):
        print(f"no {a.labels}", file=sys.stderr)
        return 1

    task = {}
    with open(a.task, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                t = json.loads(line)
                task[t["url"]] = t

    rows = []
    with open(a.labels, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    def reopen(r: dict) -> str | None:
        t = task.get(r["url"], {})
        if r["label"] and r["label"] in a.topic:
            return f"labelled {r['label']}"
        if a.skipped_substantive and not r["label"] \
                and r.get("reason") != "ambiguous":
            title = t.get("title", "")
            if BOILERPLATE not in title and t.get("n_chars", 0) >= a.min_chars:
                return "skipped but substantive"
        return None

    keep, drop = [], []
    for r in rows:
        why = reopen(r)
        (drop if why else keep).append((r, why))

    print(f"  {len(rows)} labels on file")
    print(f"  reopening {len(drop)}, keeping {len(keep)}")
    if not drop:
        print("  nothing matched")
        return 0

    by_why: dict[str, int] = {}
    for _, why in drop:
        by_why[why] = by_why.get(why, 0) + 1
    for why, n in sorted(by_why.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>4}  {why}")

    print("\n  first few:")
    for r, why in drop[:6]:
        t = task.get(r["url"], {})
        print(f"    [{why:<26}] {t.get('title','')[:52]}")

    if a.dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    backup = a.labels + f".bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(a.labels, backup)
    tmp = a.labels + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r, _ in keep:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, a.labels)

    print(f"\n  backed up to {os.path.basename(backup)}")
    print(f"  {len(keep)} labels kept; {len(drop)} articles will be served again")
    print(f"\n  next: python -m classifier.label_web")
    return 0


if __name__ == "__main__":
    sys.exit(main())
