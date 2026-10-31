"""Guard against a dataset built with a different taxonomy.

train.jsonl is a materialised artefact; taxonomy.py is read at runtime. Nothing
connects them unless something checks, so everything that consumes the dataset
calls check_dataset() first and refuses rather than reporting wrong numbers.
"""

from __future__ import annotations

import json
import os
import sys

from classifier.taxonomy import fingerprint


def check_dataset(data_dir: str) -> dict:
    path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} missing -- this dataset predates fingerprinting.\n"
              f"       Rebuild it:  python -m classifier.build_dataset "
              f"--data data-vm --out {data_dir}", file=sys.stderr)
        raise SystemExit(2)
    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    want, have = fingerprint(), man.get("taxonomy_fingerprint")
    if want != have:
        print(f"ERROR: dataset is STALE.\n"
              f"       built with taxonomy {have}\n"
              f"       taxonomy.py is now {want}\n"
              f"       Rebuild it:  python -m classifier.build_dataset "
              f"--data data-vm --out {data_dir}", file=sys.stderr)
        raise SystemExit(2)
    return man
