#!/usr/bin/env python3
"""Build the public MyHumNews release from the crawled corpus.

    .venv/bin/python dataset/build_release.py

Unions four source trees, dedupes by url, normalises the fields that are
inconsistent across outlets, attaches the DVB 19-label scheme, and writes
dataset/myhumnews.jsonl.

Every normalisation here fixes a MEASURED defect, not a hypothetical one,
and each is counted so the report says what actually changed:

  html entities   367 articles carry raw &#8220; &#8211; &amp; etc, mostly
                  in irrawaddy and myanmarnow titles
  post_id type    int for dvb, str for everyone else -- a typed Parquet
                  schema cannot hold both
  date suffix     dvb and rfa emit "Z", the others "+00:00", with varying
                  subsecond precision
  body whitespace 22,541 bodies start with whitespace
  language        irrawaddy and myanmarnow report "en-US" while their
                  bodies are 81-86% Burmese script: it is the CMS locale,
                  not the article language. Kept as cms_locale rather
                  than deleted, because it is real provenance.

The LABEL SET is dataset/dvb_scheme.py -- DVB's own editorial categories,
reduced to the 19 clearing MIN_SUPPORT=100. It is not modified here, and it
is not this project's invention: the only judgements in it are the merges
and drops, each with a recorded reason.

classifier/taxonomy.py's 10 invented classes are NO LONGER SHIPPED. They
predated the point at which the project's target was settled, and a scheme
I made up is harder to defend than the one a Burmese newsroom files
against. The raw `category` and `categories` columns are still shipped
unaltered, so anyone can rebuild that mapping or any other.
"""
from __future__ import annotations

import collections
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import dvb_scheme  # noqa: E402
from dataset import outlet_crosswalk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset"

# First occurrence of a url wins, so order matters: richest/most-canonical
# tree first. data-vm is the original crawl, data holds the complete local
# outlets, then the partial and VM-pulled backfills.
SOURCES = ["data-vm", "data", "data-backfill", "data-backfill-vm"]
OUTLETS = ["bbc", "dvb", "irrawaddy", "myanmarnow", "rfa"]

EXPECTED_TOTAL = 134_667
EXPECTED_PER_OUTLET = {
    "dvb": 78_083, "rfa": 23_534, "irrawaddy": 14_570,
    "bbc": 9_279, "myanmarnow": 9_201,
}

# ---------------------------------------------------------------------------
# Field selection. `author` is a live question, not a settled one: myanmarnow
# publishes ZERO bylines across all of its articles, which in a country where
# this very corpus reports journalists in detention reads as a deliberate
# newsroom safety choice. dvb and irrawaddy bylines are overwhelmingly the
# ORGANISATION name, but rfa and bbc carry real reporter names. Toggle here.
# ---------------------------------------------------------------------------
INCLUDE = {
    "url": True, "outlet": True, "title": True, "body": True,
    "published_date": True, "modified_date": True,
    "category": True, "categories": True, "tags": True,
    "topic_dvb19": True,
    "topic_dvb19_predicted": True,
    "topic_dvb19_margin": True,
    "author": True,          # <-- flagged for review, see note above
    "language": True, "cms_locale": True,
    "post_id": True, "content_sha256": True, "scraped_at": True,
    "section": True, "is_prose": True,
}

ENTITY_FIELDS = ("title", "body", "category")
stats: collections.Counter = collections.Counter()

# Predictions from dataset/predict_topics.py, if that pass has been run.
# The pipeline is deliberately three steps, because step 2 reads step 1's
# output:
#     build_release.py     -> publisher labels only
#     predict_topics.py    -> predictions for the unlabelled remainder
#     build_release.py     -> again, attaching them as separate columns
# Absent the file, the release simply carries no prediction columns, which is
# the correct state for a publisher-labels-only release.
PREDICTIONS: dict[str, dict] = {}
PRED_PATH = Path("dataset/dvb_topic_predictions.jsonl")
if PRED_PATH.exists():
    for _line in PRED_PATH.open():
        _d = json.loads(_line)
        PREDICTIONS[_d["url"]] = _d
    print("predictions: %s articles from %s" % (len(PREDICTIONS), PRED_PATH))
else:
    print("predictions: none (%s absent) -- publisher labels only" % PRED_PATH)


def unescape(value):
    """html.unescape, applied through lists, counting what it changes."""
    if isinstance(value, str):
        return html.unescape(value)
    if isinstance(value, list):
        return [unescape(v) for v in value]
    return value


def norm_date(value, field="date"):
    """Any ISO-8601 instant -> one canonical UTC spelling, instant preserved."""
    if not value:
        return None
    raw = value.strip()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        stats[f"dates_unparseable_{field}"] += 1
        return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    out = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if out != raw:
        stats[f"dates_renormalised_{field}"] += 1
    return out


def build_row(rec: dict) -> dict:
    row = dict(rec)

    for f in ENTITY_FIELDS:
        if f in row:
            before = row[f]
            row[f] = unescape(before)
            if isinstance(before, str) and row[f] != before:
                stats[f"entities_fixed_{f}"] += 1
    if "categories" in row:
        before = row["categories"]
        row["categories"] = unescape(before)
        if row["categories"] != before:
            stats["entities_fixed_categories"] += 1

    if row.get("post_id") is not None and not isinstance(row["post_id"], str):
        stats["post_id_cast_to_str"] += 1
        row["post_id"] = str(row["post_id"])

    for f in ("published_date", "modified_date", "scraped_at"):
        if f in row:
            row[f] = norm_date(row[f], f)

    body = row.get("body")
    if isinstance(body, str):
        stripped = body.strip()
        if stripped != body:
            stats["bodies_whitespace_stripped"] += 1
        if body[:1].isspace():
            stats["bodies_leading_whitespace"] += 1
        if body[-1:].isspace():
            stats["bodies_trailing_whitespace"] += 1
        row["body"] = stripped
    title = row.get("title")
    if isinstance(title, str):
        row["title"] = title.strip()

    # The crawled `language` is the publishing CMS's locale, which for two
    # outlets disagrees with the actual script. Preserve it under an honest
    # name and state the real language, which is uniform across the corpus.
    row["cms_locale"] = row.pop("language", None)
    row["language"] = "my"

    # `author` arrives as a list on 12 myanmarnow rows (always empty -- that
    # outlet publishes no bylines), which breaks a typed schema. Collapse to
    # a nullable string.
    a = row.get("author")
    if isinstance(a, list):
        stats["author_list_coerced"] += 1
        row["author"] = "; ".join(x for x in a if x) or None

    # `tags` is absent for four of five outlets; make it an empty list so the
    # column is list<string> everywhere rather than mixed null/list.
    if row.get("tags") is None:
        row["tags"] = []
    if row.get("categories") is None:
        row["categories"] = []

    cats = row.get("categories") or ([row["category"]] if row.get("category") else [])
    # The release's label set is DVB's own editorial categories, reduced to the
    # 19 that clear MIN_SUPPORT. classifier/taxonomy.py's 10 invented classes
    # are deprecated and no longer shipped: they predated the project's target
    # and were my scheme rather than a newsroom's.
    #
    # Non-DVB outlets receive labels only where their OWN category vocabulary
    # coincides with DVB's -- BBC files under နိုင်ငံရေး too. Those are still
    # publisher-assigned, by a different newsroom, and the `outlet` column is
    # how a user restricts to DVB if they want one editorial convention.
    # outlet_crosswalk.labels() is dvb_scheme's 19 PLUS the other outlets'
    # equivalent categories, plus the human_rights label those outlets have
    # and DVB does not. Every label it returns is still publisher-assigned:
    # the crosswalk translates vocabularies, it does not infer anything.
    row["topic_dvb19"] = outlet_crosswalk.labels(cats)
    if row["topic_dvb19"]:
        stats["topic_assigned"] += 1
    else:
        stats["topic_none"] += 1

    # Predictions live in their OWN column, never merged into topic_dvb19.
    # Someone who loads this and ignores a column should get publisher labels
    # only -- not a silent mixture including labels that transfer at 0.645
    # precision across outlets. Opting in is one explicit line:
    #     labels = row["topic_dvb19"] or row["topic_dvb19_predicted"]
    pred = PREDICTIONS.get(row["url"])
    if pred and not row["topic_dvb19"]:
        labs = sorted(pred["topic_dvb19"])
        row["topic_dvb19_predicted"] = labs
        # margins are positionally aligned with the labels above
        row["topic_dvb19_margin"] = [pred["margins"][l] for l in labs]
        if labs:
            stats["topic_predicted"] += 1
    else:
        row["topic_dvb19_predicted"] = []
        row["topic_dvb19_margin"] = []

    return {k: row.get(k) for k, v in INCLUDE.items() if v}


def write_parquet(order: list[str], seen: dict) -> None:
    """Parquet with an EXPLICIT schema.

    Left to inference, pyarrow would read `tags` as null-typed where a
    leading run is empty, which then fails to merge, and `topic_dvb19` as
    list<null> for a chunk of unlabelled rows. Stating the schema makes the
    file portable.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("\npyarrow not installed -- skipping Parquet, JSONL is the release")
        return

    S = pa.string()
    schema = pa.schema([
        ("url", S), ("outlet", S), ("title", S), ("body", S),
        ("published_date", S), ("modified_date", S),
        ("category", S), ("categories", pa.list_(S)), ("tags", pa.list_(S)),
        ("topic_dvb19", pa.list_(S)),
        ("topic_dvb19_predicted", pa.list_(S)),
        ("topic_dvb19_margin", pa.list_(pa.float32())),
        ("author", S),
        ("language", S), ("cms_locale", S),
        ("post_id", S), ("content_sha256", S), ("scraped_at", S),
        ("section", S), ("is_prose", pa.bool_()),
    ])
    cols = {f.name: [] for f in schema}
    for url in order:
        r = seen[url]
        for f in schema:
            v = r.get(f.name)
            if f.name in ("categories", "tags") and v is None:
                v = []
            cols[f.name].append(v)
    table = pa.table({k: pa.array(v, type=schema.field(k).type)
                      for k, v in cols.items()}, schema=schema)
    dest = OUT / "myhumnews.parquet"
    pq.write_table(table, dest, compression="zstd")
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.0f} MB, zstd)")


def main() -> None:
    seen: dict[str, dict] = {}
    per_source: collections.Counter = collections.Counter()
    dupes = 0

    for base in SOURCES:
        for outlet in OUTLETS:
            path = ROOT / base / outlet / "parsed" / "articles.jsonl"
            if not path.exists():
                continue
            for line in path.open():
                rec = json.loads(line)
                url = rec.get("url")
                if not url:
                    stats["rows_without_url"] += 1
                    continue
                if url in seen:
                    dupes += 1
                    continue
                seen[url] = build_row(rec)
                per_source[base] += 1

    per_outlet = collections.Counter(r["outlet"] for r in seen.values())
    # The crawl stores an outlet slug ("bbc_burmese"); map back for the check.
    slug = {"bbc_burmese": "bbc", "dvb_burmese": "dvb",
            "irrawaddy_burmese": "irrawaddy",
            "myanmar_now_burmese": "myanmarnow", "rfa_burmese": "rfa"}
    counted = collections.Counter()
    for k, v in per_outlet.items():
        counted[slug.get(k, k)] += v

    print("=== UNION ===")
    print(f"  rows read (pre-dedupe) : {len(seen) + dupes}")
    print(f"  duplicate urls dropped : {dupes}")
    print(f"  UNIQUE ARTICLES        : {len(seen)}")
    print("\n  first contributed by source:")
    for k in SOURCES:
        print(f"    {k:<18} {per_source[k]:>7}")
    print("\n  per outlet (expected -> actual):")
    ok = True
    for o, exp in EXPECTED_PER_OUTLET.items():
        got = counted.get(o, 0)
        mark = "ok" if got == exp else "MISMATCH"
        if got != exp:
            ok = False
        print(f"    {o:<12} {exp:>7} -> {got:>7}  {mark}")

    if len(seen) != EXPECTED_TOTAL or not ok:
        print(f"\nSTOPPING: expected {EXPECTED_TOTAL} unique articles, got {len(seen)}.")
        print("Not writing output. The union or the sources have changed.")
        sys.exit(1)

    print("\n=== NORMALISATION (rows affected) ===")
    for k in sorted(stats):
        print(f"  {k:<32} {stats[k]:>7}")

    OUT.mkdir(exist_ok=True)
    dest = OUT / "myhumnews.jsonl"
    with dest.open("w") as f:
        for url in sorted(seen):
            f.write(json.dumps(seen[url], ensure_ascii=False) + "\n")
    print(f"\nwrote {dest} ({dest.stat().st_size / 1e6:.0f} MB)")

    write_parquet(sorted(seen), seen)

    years = collections.Counter(
        (r["published_date"] or "")[:4] for r in seen.values())
    print("\nby year:", dict(sorted(years.items())))
    if stats["topic_predicted"]:
        pub, pre, n = stats["topic_assigned"], stats["topic_predicted"], len(seen)
        print("topic_dvb19_predicted:  %d articles (%.1f%%)" % (pre, 100 * pre / n))
        print("combined coverage:      %d of %d (%.1f%%)"
              % (pub + pre, n, 100 * (pub + pre) / n))
    print("topic_dvb19 coverage: %d of %d (%.1f%%)"
          % (stats["topic_assigned"], len(seen),
             100 * stats["topic_assigned"] / len(seen)))
    print("label scheme: dvb_scheme %d labels + crosswalk (%d mappings, "
          "%d recorded omissions) = %d labels, MIN_SUPPORT=%d"
          % (len(dvb_scheme.F1_TEMPORAL), len(outlet_crosswalk.CROSSWALK),
             len(outlet_crosswalk.NOT_MAPPED),
             len(set(dvb_scheme.F1_TEMPORAL) |
                 {v[0] for v in outlet_crosswalk.CROSSWALK.values()}),
             dvb_scheme.MIN_SUPPORT))


if __name__ == "__main__":
    main()
