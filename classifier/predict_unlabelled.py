#!/usr/bin/env python3
"""
Assign topics to the corpus articles that carry no usable outlet label.

    python -m classifier.predict_unlabelled --data data-vm

29,203 of 38,152 articles have an outlet label that maps to a topic. The rest
do not -- including ALL of RFA (its labels are programme names and section
furniture) and nearly all of Myanmar Now. Those outlets are therefore absent
from any analysis keyed on topic, which defeats the point of a five-outlet
corpus. This script fills them in with the model's prediction.

A predicted label is not a outlet label and this script never pretends otherwise:
every row is stamped source="predicted" with the margin that produced it, and
the two layers stay distinguishable downstream.

Two fits, deliberately:

  1. fit on train.jsonl, predict test.jsonl   -> the margin/accuracy table.
     LinearSVC has no probabilities, so the confidence proxy is the gap between
     the best and second-best class score. The table says what that gap is
     actually worth, measured on held-out data, so a threshold can be chosen
     from evidence rather than taste.

  2. refit on train + test, predict the unlabelled.
     For labelling you want every example you have. The quality estimate comes
     from fit 1; fit 2 is strictly better-informed than the model that estimate
     describes, so the estimate is conservative.

CAVEAT: the held-out accuracy was measured on DVB, BBC and Irrawaddy, the
outlets that have labels. Most of what gets predicted here is RFA and Myanmar
Now, which the model has barely seen (RFA: zero training examples), so the
headline number does not transfer to them. A 24-article spot-check across three
margin bands suggests the margin ORDERING does transfer -- high-margin
predictions were right, low-margin ones held the errors -- so filter on margin
rather than trusting the per-article label. classifier/data/spotcheck.jsonl
holds that sample; extend it before leaning on these labels in a write-up.

MIN_TRAIN_PER_SITE -- why an outlet can be excluded from training:
Myanmar Now has 40 labelled articles, 35 of them economy. That is not a sample
of Myanmar Now's output, it is the handful of its articles that happened to
carry a mappable label, and the class skew is an artefact of which labels map.
Training on them taught the model "Myanmar Now house style => economy": it then
predicted economy for 27.5% of Myanmar Now's unlabelled articles against a 6.9%
corpus prior, a 4x over-fire seen in no other outlet. Dropping those 40 moved
economy to 8.5% AND raised the median margin from 0.62 to 0.84 -- more
confident on less data, which is the signature of removing contradictory signal
rather than removing information. So: an outlet with too few labelled articles
to represent itself is better left out of training entirely than in as a biased
prior on its own style.

Outputs:
  classifier/data/predicted.jsonl  one row per newly-labelled article
  classifier/data/topics.jsonl     url -> topic for ALL articles, outlet and
                                   predicted together, each tagged by source
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from classifier.dataset_guard import check_dataset  # noqa: E402
from classifier.taxonomy import label_article  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SITES = ["dvb", "bbc", "rfa", "irrawaddy", "myanmarnow"]
MAX_CHARS = 3000
MIN_CHARS = 200
# Below this many labelled articles, an outlet is dropped from TRAINING (it is
# still predicted). See the note above -- measured, not guessed.
MIN_TRAIN_PER_SITE = 200


def norm_text(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def feats(title: str, text: str) -> str:
    return (title + "\n" + text)[:MAX_CHARS]


def load_split(path: str):
    X, y, meta = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            X.append(feats(r["title"], r["text"]))
            y.append(r["topic"])
            meta.append(r)
    return X, y, meta


def build_pipeline() -> object:
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                        max_features=300_000, min_df=3, sublinear_tf=True),
        LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=5000),
    )


def margins(pipe, X) -> tuple[np.ndarray, np.ndarray]:
    """Predicted class index and the top1-top2 score gap for each row."""
    scores = pipe.decision_function(X)
    order = np.argsort(scores, axis=1)
    top = order[:, -1]
    second = order[:, -2]
    rows = np.arange(scores.shape[0])
    gap = scores[rows, top] - scores[rows, second]
    return top, gap


def margin_table(pipe, X, y_true) -> list[tuple[float, float, int, float]]:
    """Held-out accuracy as a function of the margin threshold."""
    top, gap = margins(pipe, X)
    classes = pipe.named_steps["linearsvc"].classes_
    pred = classes[top]
    correct = np.array([p == t for p, t in zip(pred, y_true)])

    out = []
    for thr in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5):
        keep = gap >= thr
        n = int(keep.sum())
        if not n:
            continue
        out.append((thr, float(correct[keep].mean()), n, n / len(gap)))
    return out


def iter_corpus(data_root: str):
    for site in SITES:
        path = os.path.join(data_root, site, "parsed", "articles.jsonl")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield site, json.loads(line)
                except json.JSONDecodeError:
                    continue


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data-vm", help="corpus root")
    p.add_argument("--dataset", default=os.path.join(HERE, "data"))
    p.add_argument("--batch", type=int, default=2000,
                   help="articles vectorised at once; keeps memory flat")
    p.add_argument("--min-train-per-site", type=int, default=MIN_TRAIN_PER_SITE,
                   help="outlets with fewer labelled articles are excluded "
                        "from training; 0 disables")
    a = p.parse_args()

    # ---- fit 1: the margin/accuracy table ----
    print("=== fit 1: measuring what the margin is worth ===")
    check_dataset(a.dataset)
    Xtr, ytr, train_meta = load_split(os.path.join(a.dataset, "train.jsonl"))
    Xte, yte, test_meta = load_split(os.path.join(a.dataset, "test.jsonl"))
    print(f"  train {len(Xtr)}  test {len(Xte)}")

    # Exclude outlets whose labelled sample is too small to represent them.
    counts = collections.Counter(r["site"] for r in train_meta + test_meta)
    thin = {s for s, n in counts.items() if n < a.min_train_per_site}
    if thin:
        for s in sorted(thin):
            print(f"  EXCLUDED from training: {s} ({counts[s]} labelled, "
                  f"< {a.min_train_per_site}) -- still predicted")

    def sieve(X, y, meta):
        z = [(x, t) for x, t, m in zip(X, y, meta) if m["site"] not in thin]
        return [x for x, _ in z], [t for _, t in z]

    Xtr, ytr = sieve(Xtr, ytr, train_meta)
    Xte_f, yte_f = sieve(Xte, yte, test_meta)
    print(f"  after exclusion: train {len(Xtr)}  test {len(Xte_f)}")

    t0 = time.time()
    pipe = build_pipeline().fit(Xtr, ytr)
    print(f"  fit {time.time() - t0:.0f}s")

    top, _ = margins(pipe, Xte_f)
    classes = pipe.named_steps["linearsvc"].classes_
    pred_te = classes[top]
    print(f"  held-out accuracy {accuracy_score(yte_f, pred_te):.3f}  "
          f"macro-F1 {f1_score(yte_f, pred_te, average='macro', zero_division=0):.3f}")

    print(f"\n  {'margin>=':>9} {'accuracy':>9} {'kept':>8} {'coverage':>9}")
    for thr, acc, n, cov in margin_table(pipe, Xte_f, yte_f):
        print(f"  {thr:>9.2f} {acc:>9.3f} {n:>8} {cov:>8.1%}")

    # ---- fit 2: the model that does the labelling ----
    print("\n=== fit 2: refitting on train + test ===")
    t0 = time.time()
    pipe = build_pipeline().fit(Xtr + Xte_f, ytr + yte_f)
    print(f"  fit {time.time() - t0:.0f}s on {len(Xtr) + len(Xte_f)} articles")
    classes = pipe.named_steps["linearsvc"].classes_

    # Outlet labels, keyed by url, so topics.jsonl can carry both layers.
    # These are the outlets' own section names mapped by taxonomy.py --
    # distant supervision, not verified ground truth. See README.
    outlet_lab = {r["url"]: r["topic"] for r in test_meta}
    with open(os.path.join(a.dataset, "train.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                outlet_lab[r["url"]] = r["topic"]

    # ---- stream the corpus, predict in batches ----
    print("\n=== predicting ===")
    pred_path = os.path.join(a.dataset, "predicted.jsonl")
    topics_path = os.path.join(a.dataset, "topics.jsonl")
    stats = collections.Counter()
    by_site = collections.defaultdict(collections.Counter)
    seen_urls: set[str] = set()

    buf: list[dict] = []

    def flush(fh_pred, fh_top):
        if not buf:
            return
        top_i, gap = margins(pipe, [b["_x"] for b in buf])
        for b, ti, g in zip(buf, top_i, gap):
            topic = classes[ti]
            row = {
                "url": b["url"], "site": b["site"], "outlet": b["outlet"],
                "published_date": b["published_date"], "title": b["title"],
                "topic": topic, "margin": round(float(g), 4),
                "n_chars": b["n_chars"], "short": b["n_chars"] < MIN_CHARS,
                "raw_labels": b["raw_labels"],
            }
            fh_pred.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh_top.write(json.dumps({
                "url": b["url"], "site": b["site"],
                "published_date": b["published_date"], "topic": topic,
                "source": "predicted", "margin": round(float(g), 4),
            }, ensure_ascii=False) + "\n")
            by_site[b["site"]][topic] += 1
            stats["predicted"] += 1
        buf.clear()

    with open(pred_path, "w", encoding="utf-8") as fh_pred, \
            open(topics_path, "w", encoding="utf-8") as fh_top:
        for site, r in iter_corpus(a.data):
            stats["read"] += 1
            url = r.get("url")
            if not url or url in seen_urls:
                stats["dup_url"] += 1
                continue
            seen_urls.add(url)

            if url in outlet_lab:
                fh_top.write(json.dumps({
                    "url": url, "site": site,
                    "published_date": r.get("published_date"),
                    "topic": outlet_lab[url], "source": "outlet", "margin": None,
                }, ensure_ascii=False) + "\n")
                stats["outlet"] += 1
                by_site[site][outlet_lab[url]] += 0  # register the site
                continue

            body = norm_text(r.get("body"))
            if not body.strip():
                stats["empty"] += 1
                continue

            labels = r.get("categories") or (
                [r["category"]] if r.get("category") else [])
            # Sanity check: anything the taxonomy CAN label should have been
            # caught by build_dataset. If this fires, the two are out of sync.
            if label_article(labels) is not None:
                stats["labelable_but_not_in_dataset"] += 1

            title = norm_text(r.get("title"))
            buf.append({
                "url": url, "site": site, "outlet": r.get("outlet") or site,
                "published_date": r.get("published_date"), "title": title,
                "n_chars": len(body), "raw_labels": labels,
                "_x": feats(title, body),
            })
            if len(buf) >= a.batch:
                flush(fh_pred, fh_top)
        flush(fh_pred, fh_top)

    # ---- report ----
    print("\n=== coverage ===")
    for k in ("read", "dup_url", "empty", "outlet", "predicted",
              "labelable_but_not_in_dataset"):
        if stats[k]:
            print(f"  {k:30} {stats[k]:>7}")
    total = stats["outlet"] + stats["predicted"]
    print(f"  {'TOTAL with a topic':30} {total:>7}")

    print("\n=== predicted topics by outlet ===")
    sites = [s for s in SITES if by_site[s]]
    hdr = f"  {'topic':14}" + "".join(f"{s[:9]:>11}" for s in sites) + f"{'all':>9}"
    print(hdr)
    topics = sorted({t for c in by_site.values() for t in c},
                    key=lambda t: -sum(by_site[s][t] for s in sites))
    for t in topics:
        tot = sum(by_site[s][t] for s in sites)
        print(f"  {t:14}" + "".join(f"{by_site[s][t]:>11}" for s in sites)
              + f"{tot:>9}")

    print(f"\nwrote {pred_path}")
    print(f"wrote {topics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
