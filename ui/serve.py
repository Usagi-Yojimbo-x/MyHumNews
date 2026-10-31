#!/usr/bin/env python3
"""
Cross-lingual search over the Burmese corpus: ask in English, get Burmese news.

    .venv/bin/python -m ui.serve

Then open http://127.0.0.1:8770 -- it prints the URL.

WHY THIS EXISTS: the project's premise is reading Myanmar in Burmese for people
who do not read Burmese. An English question is embedded into the same vector
space as Burmese article text by BGE-M3, so retrieval crosses the language
boundary with no translation step anywhere. This page is where that claim is
visible rather than asserted.

WHY A LOCAL SERVER AND NOT A PUBLISHED PAGE (same reasoning as label_web):
the index is 156MB of float32 vectors and every query needs BGE-M3 on MPS to
encode it. Neither travels to a browser. It also keeps the corpus on this
machine -- there is no reason to ship 38,092 news articles to a third party.
Binds 127.0.0.1 only.

SCORING -- the hybrid, not dense alone:

    score = cosine(query, article) + ALPHA * gazetteer_hit

Measured on the 29 known-item questions: dense alone gives Recall@1 0.207,
dense + gazetteer gives 0.276, leave-one-out. The reason the lexical channel
is needed at all is that BGE-M3 aligns *translated* names well (National Unity
Government <-> အမျိုးသားညီညွတ်ရေးအစိုးရ, cosine 0.790) but barely aligns
*transliterated* ones -- place names 0.487, and `Mandalay` scores closer to
`နေပြည်တော်` (Naypyitaw) than to `မန္တလေး`. Every known-item miss hinged on a
transliterated name.

The scoring logic is COPIED from rag/eval/hybrid.py rather than imported, on
purpose: that file is an experiment under active change, and a UI that breaks
when an ablation is being run is worse than a UI with twelve duplicated lines.
rag/index/, rag/gazetteer/ and rag/eval/ are read-only inputs here.

WHY THE GAZETTEER MATCHES A SUFFIXED FORM (ဆောမြို့, never bare ဆော):
Burmese has no word boundaries, so a short name is a substring of common
words. `ဆော` alone occurs 117,574 times in this corpus; `ဆောမြို့` -- Saw
township -- occurs 418. `ကလေး` (Kalay) also means "child"; `ပေါက်` (Pauk) also
means "to burst". Documents are whitespace-stripped before matching because
Burmese spaces words inconsistently, so one entity has several surface forms.

STARTUP COST, PAID ONCE: the gazetteer holds 54 entries, so instead of
scanning 38,092 bodies per query, a 38,092 x 54 boolean hit matrix is built at
load time. A query then ORs a few columns, which is free. This is why the first
run takes a minute and every query after it is fast.
"""

from __future__ import annotations

import argparse
import glob
import http.server
import json
import os
import re
import socketserver
import sys
import threading
import time
import unicodedata
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rag import embed  # noqa: E402

WS = re.compile(r"\s+")


def flatten(text: str) -> str:
    """Whitespace-stripped and NFC-normalised, for substring matching.

    Both steps are required and each was measured on this corpus:

      whitespace -- Burmese spaces words inconsistently, so one entity has
      several surface forms (TNLA appears three ways).

      NFC -- 6,991 of the 38,092 indexed articles (18%) are NOT NFC
      normalised. Burmese combining marks can appear in either order and
      render identically, so the same word yields different codepoint
      sequences and a substring test silently misses. Normalising both sides
      moves 34 gazetteer hits on that subset. Small, but it is free, and a
      miss here is invisible rather than noisy.
    """
    return unicodedata.normalize("NFC", WS.sub("", text))
ALPHA = 0.10          # leave-one-out optimum; result is flat from 0.05 to 0.5
SNIPPET = 700         # chars of body shown on a result card
GEN_SNIPPET = 1800    # chars handed to the generator. MUST match what
                      # rag/eval/eval_answers.py measured, or the demo
                      # behaves differently from the published numbers.


def load_examples(path: str, n: int = 6) -> list[dict]:
    """The left pane's example queries, read from the evaluation set.

    These are the actual known-item questions the retrieval numbers were
    measured on, taken in FILE ORDER and not cherry-picked. That matters:
    K01-K06 happen to include K05, which the system misses badly (rank 87).
    A demo built only from the wins would misrepresent the system, and the
    eval set is the honest place to draw from -- it doubles as a window onto
    what was actually measured.
    """
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("type") == "known_item":
            out.append({"qid": r["qid"], "question": r["question"]})
        if len(out) >= n:
            break
    return out


HUM: dict[str, dict] = {}


def load_humanitarian(path: str) -> int:
    """Predicted HumSet labels from classifier/predict_corpus.py, by url.

    Two of the seven labels are deliberately NOT presented as findings.
    `humanitarian_access` rests on 16 gold positives with kappa 0.091 between
    annotators -- no agreed human boundary -- and `information_communication`
    on 6. predict_corpus.py writes them but excludes them from
    `labels_filter`, which is the field a retrieval filter should use. This
    keeps that separation: the trusted set is shown as labels, the other two
    only ever as explicitly low-confidence. Collapsing the two fields into one
    list here would quietly undo the one safeguard that script exists to
    provide.
    """
    HUM.clear()
    if not os.path.exists(path):
        return 0
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        HUM[d["url"]] = {"labels": d.get("labels_filter") or [],
                         "low": d.get("low_confidence") or []}
    return len(HUM)


def humanitarian_labels(url: str) -> dict:
    """Trusted labels and low-confidence ones, kept apart on purpose."""
    return HUM.get(url) or {"labels": [], "low": []}


class Index:
    """Owns the vectors, the per-article display data and the hit matrix."""

    def __init__(self, index_dir: str, gaz_path: str, data_glob: str,
                 snippet_chars: int = SNIPPET, cache_dir: str | None = None):
        t0 = time.time()
        self.vecs = np.load(os.path.join(index_dir, "vectors.npy"))
        self.meta = [json.loads(l) for l in
                     open(os.path.join(index_dir, "meta.jsonl"), encoding="utf-8")]
        if len(self.meta) != len(self.vecs):
            raise SystemExit(f"index mismatch: {len(self.meta)} meta rows "
                             f"vs {len(self.vecs)} vectors")
        self.manifest = {}
        mf = os.path.join(index_dir, "manifest.json")
        if os.path.exists(mf):
            self.manifest = json.load(open(mf, encoding="utf-8"))
        self.gaz = json.load(open(gaz_path, encoding="utf-8"))
        self.n_indexed = len(self.meta)
        # Stable column order so the hit matrix is reproducible.
        self.places = sorted(self.gaz)
        self.forms = [flatten(self.gaz[p]["match"]) for p in self.places]
        print(f"  index   {len(self.vecs)} articles, dim {self.vecs.shape[1]}")
        print(f"  gazetteer {len(self.places)} verified place names")

        if cache_dir and self._load_cache(cache_dir, snippet_chars):
            print(f"  loaded in {time.time() - t0:.1f}s (from cache)")
            return
        self._build(data_glob, snippet_chars)
        if cache_dir:
            self._save_cache(cache_dir, snippet_chars)
        print(f"  loaded in {time.time() - t0:.1f}s")

    # The cache exists so the server can run on a host WITHOUT the 1.5GB
    # corpus: snippets plus hit matrix are all the corpus pass produces. Its
    # key is everything the pass depends on -- the indexed urls, the match
    # forms and the snippet length -- so a changed gazetteer or a rebuilt
    # index rebuilds the cache instead of silently serving a stale one.
    def _cache_key(self, snippet_chars: int) -> dict:
        return {"urls": len(self.meta),
                "first_url": self.meta[0]["url"], "last_url": self.meta[-1]["url"],
                "forms": self.forms, "snippet_chars": snippet_chars}

    def _load_cache(self, cache_dir: str, snippet_chars: int) -> bool:
        kf = os.path.join(cache_dir, "key.json")
        if not os.path.exists(kf):
            return False
        if json.load(open(kf, encoding="utf-8")) != self._cache_key(snippet_chars):
            print("  cache   stale -- rebuilding from the corpus")
            return False
        self.hits = np.load(os.path.join(cache_dir, "hits.npy"))
        self.snip = [json.loads(l) for l in
                     open(os.path.join(cache_dir, "snippets.jsonl"), encoding="utf-8")]
        print(f"  hit matrix {self.hits.shape} -- "
              f"{int(self.hits.sum())} (article, place) pairs")
        return True

    def _save_cache(self, cache_dir: str, snippet_chars: int) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(os.path.join(cache_dir, "hits.npy"), self.hits)
        with open(os.path.join(cache_dir, "snippets.jsonl"), "w",
                  encoding="utf-8") as f:
            for s in self.snip:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        json.dump(self._cache_key(snippet_chars),
                  open(os.path.join(cache_dir, "key.json"), "w", encoding="utf-8"),
                  ensure_ascii=False)
        print(f"  cache   written to {cache_dir}")

    def _build(self, data_glob: str, snippet_chars: int) -> None:
        row = {m["url"]: i for i, m in enumerate(self.meta)}
        self.snip: list[str] = [""] * len(self.meta)
        self.hits = np.zeros((len(self.meta), len(self.places)), dtype=bool)

        # One pass over the corpus: build the hit matrix, keep a snippet, and
        # drop the full body. Keeping 38,092 whole articles in RAM to re-scan
        # them on every query would cost ~100MB for no benefit. The answer
        # evaluation passes a larger snippet_chars, because a generator needs
        # more of the article than a search result card does.
        seen = 0
        for f in sorted(glob.glob(data_glob)):
            for line in open(f, encoding="utf-8"):
                d = json.loads(line)
                i = row.get(d["url"])
                if i is None:
                    continue
                seen += 1
                title = d.get("title") or ""
                body = d.get("body") or ""
                self.snip[i] = body[:snippet_chars]
                flat = flatten(title + body)
                for c, form in enumerate(self.forms):
                    if form in flat:
                        self.hits[i, c] = True
        print(f"  bodies  matched {seen}/{len(self.meta)} indexed articles")
        print(f"  hit matrix {self.hits.shape} -- "
              f"{int(self.hits.sum())} (article, place) pairs")

    def matched_places(self, question: str) -> list[int]:
        """Gazetteer entries named in the query, as hit-matrix columns.

        Word-boundary match on the English side: a bare substring test would
        fire `Mon` inside "Monday" and `Saw` inside "sawmill".
        """
        out = []
        for c, en in enumerate(self.places):
            if re.search(rf"\b{re.escape(en)}\b", question, re.I):
                out.append(c)
        return out

    def search(self, question: str, k: int = 20, alpha: float = ALPHA) -> dict:
        qvec = embed.encode([question], max_len=64, batch=1)[0]
        dense = self.vecs @ qvec                       # cosine: both L2-normed
        cols = self.matched_places(question)
        if cols:
            lex = self.hits[:, cols].any(axis=1)
        else:
            lex = np.zeros(len(dense), dtype=bool)
        score = dense + alpha * lex

        order = np.argsort(-score)[:k]
        results = []
        for rank, i in enumerate(order, 1):
            m = self.meta[i]
            results.append({
                "rank": rank,
                "score": round(float(score[i]), 4),
                "dense": round(float(dense[i]), 4),
                "bonus": round(alpha * float(lex[i]), 4),
                "lexical": bool(lex[i]),
                "url": m["url"],
                "title": m.get("title") or "",
                "outlet": m.get("outlet") or "",
                "published": (m.get("published_date") or "")[:10],
                "category": m.get("category") or "",
                "i": int(i),
                "snippet": self.snip[i][:SNIPPET],
                "humanitarian": humanitarian_labels(m["url"]),
            })
        return {
            "question": question,
            "alpha": alpha,
            "n_indexed": len(self.meta),
            "matched": [{"english": self.places[c],
                         "burmese": self.gaz[self.places[c]]["burmese"],
                         "match_form": self.gaz[self.places[c]]["match"],
                         "n_docs": int(self.hits[:, c].sum())} for c in cols],
            "n_lexical": int(lex.sum()),
            "results": results,
        }


def make_handler(index: Index, page_path: str):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):          # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj) -> None:
            self._send(200, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                # Read per request so editing the HTML needs no restart --
                # reloading the 156MB index to change a colour would be absurd.
                body = open(page_path, encoding="utf-8").read().encode("utf-8")
                self._send(200, body, "text/html; charset=utf-8")
            elif path == "/meta":
                # The manifest records the local build path ("data-vm") and
                # the build machine's device. Neither belongs in a browser
                # payload -- the page says "5 Myanmar news outlets" instead.
                pub = {k: v for k, v in index.manifest.items()
                       if k not in ("data", "device")}
                self._json({
                    "n_indexed": index.n_indexed,
                    "manifest": pub,
                    "n_places": len(index.places),
                    "places": index.places,
                    "alpha": ALPHA,
                    "examples": index.examples,
                })
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send(400, b"bad json", "text/plain")
                return
            if self.path == "/answer":
                # Generation is optional: the demo must still search when no
                # GEMINI_API_KEY is set, so this endpoint fails soft rather
                # than taking the server down at import time.
                q = (payload.get("q") or "").strip()
                if not q:
                    self._send(400, b"empty query", "text/plain")
                    return
                k = min(int(payload.get("k") or 8), 20)
                t0 = time.time()
                hit = index.search(q, k=k, alpha=ALPHA)
                # index.snip holds GEN_SNIPPET chars; r["snippet"] is the
                # shortened display copy, so read the full one by row index.
                passages = [{"title": r["title"], "text": index.snip[r["i"]],
                             "url": r["url"], "outlet": r["outlet"],
                             "published": r["published"]}
                            for r in hit["results"]]
                try:
                    from rag.generate import answer as _answer
                    a = _answer(q, passages)
                except SystemExit as e:          # no API key configured
                    self._json({"error": str(e), "results": hit["results"]})
                    return
                except Exception as e:
                    self._json({"error": "%s: %s" % (type(e).__name__, e),
                                "results": hit["results"]})
                    return
                a["results"] = hit["results"]
                a["took_ms"] = int(1000 * (time.time() - t0))
                self._json(a)
            elif self.path == "/search":
                q = (payload.get("q") or "").strip()
                if not q:
                    self._send(400, b"empty query", "text/plain")
                    return
                k = min(int(payload.get("k") or 20), 50)
                try:
                    alpha = float(payload.get("alpha", ALPHA))
                except (TypeError, ValueError):
                    alpha = ALPHA
                t0 = time.time()
                out = index.search(q, k=k, alpha=alpha)
                out["took_ms"] = int(1000 * (time.time() - t0))
                self._json(out)
            else:
                self._send(404, b"not found", "text/plain")

    return H


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index", default=os.path.join(ROOT, "rag", "index"))
    p.add_argument("--gazetteer",
                   default=os.path.join(ROOT, "rag", "gazetteer", "places.json"))
    p.add_argument("--data", default=os.path.join(ROOT, "data-vm",
                                                  "*", "parsed", "articles.jsonl"),
                   help="must be the corpus the index was built from")
    p.add_argument("--cache", default=os.path.join(ROOT, "rag", "serve_cache"),
                   help="snippets + hit matrix; lets a host without the corpus "
                        "serve. Pass '' to disable.")
    p.add_argument("--page", default=os.path.join(HERE, "search.html"))
    p.add_argument("--questions",
                   default=os.path.join(ROOT, "rag", "eval", "questions.jsonl"),
                   help="eval set the left-pane examples are drawn from")
    p.add_argument("--humanitarian",
                   default=os.path.join(ROOT, "classifier", "data",
                                        "corpus_labels.jsonl"),
                   help="output of classifier.predict_corpus")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-open", action="store_true")
    a = p.parse_args()

    for path, hint in ((os.path.join(a.index, "vectors.npy"),
                        "build it: .venv/bin/python -m rag.build_index"),
                       (a.gazetteer, "rag/gazetteer/places.json is missing"),
                       (a.page, "ui/search.html is missing from the repo")):
        if not os.path.exists(path):
            print(f"missing {path}\n  -> {hint}", file=sys.stderr)
            return 1

    print("loading -- BGE-M3 and a 156MB index, so this takes a minute")
    index = Index(a.index, a.gazetteer, a.data, snippet_chars=GEN_SNIPPET,
                  cache_dir=a.cache or None)
    index.examples = load_examples(a.questions)
    n_hum = load_humanitarian(a.humanitarian)
    print(f"  humset  {n_hum} articles with predicted labels"
          if n_hum else
          f"  humset  no predictions at {a.humanitarian} -- cards will say so")
    print(f"  examples  {len(index.examples)} known-item questions "
          f"({', '.join(e['qid'] for e in index.examples)})")
    embed.load()                     # once, at startup, never per request

    class S(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    url = f"http://{a.host}:{a.port}"
    with S((a.host, a.port), make_handler(index, a.page)) as httpd:
        cov = index.manifest.get("data", "?")
        print(f"\n  corpus {cov} -- {len(index.meta)} articles, "
              f"{index.manifest.get('min_chars', '?')}+ chars")
        print(f"  ask in English; the articles are Burmese and are NOT translated")
        print(f"\n  open {url}")
        print("  Ctrl-C to stop\n")
        if not a.no_open:
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
