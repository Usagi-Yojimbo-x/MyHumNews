#!/usr/bin/env python3
"""English answers from Burmese passages — the generation half of the RAG.

    export GEMINI_API_KEY=...                       # free tier, aistudio.google.com
    .venv/bin/python rag/generate.py --list-models  # what your key can actually call
    .venv/bin/python rag/generate.py --demo         # canned passages, no index load

Retrieval was finished and measured first (`rag/eval/RETRIEVAL_RESULTS.md`):
R@10 0.586 with the gazetteer channel, and a ceiling that nine lexical variants
and a chunked index could not move. This module is the other half — turning
retrieved Burmese passages into an English answer for a reader who does not
read Burmese, which is the entire point of the project.

## Refusal is a generation problem, and that is a measured finding

The retrieval evaluation established that a cosine threshold CANNOT separate
"nothing here answers this" from "here is the answer": known-item top-1
similarity ran min 0.487 / median 0.572 / max 0.666, negative questions ran
min 0.513 / median 0.536 / max 0.647 — completely overlapping. "When does the
next SpaceX Starship launch?" scored 0.647, beating 24 of the 29 real
questions, and it was not even a wrong retrieval: the corpus genuinely carries
a Burmese article about a Starship launch explosion.

So the generator must decide whether the passages support an answer, and this
module asks it to do exactly that: `supported` is returned as a separate field
before any answer text, and `rag/eval/eval_answers.py` scores it against the 5
negative questions.

## Citations are by passage index, on purpose

The model is given numbered passages and must cite `[2]`, not a URL. URLs can
be hallucinated plausibly; an index outside 1..n is mechanically detectable.
The indices are mapped back to URLs here, after validation.

## PRIVACY NOTE, read before running this on anything but the demo

This sends Burmese article text to Google's Generative Language API. On the
FREE tier, inputs may be used to improve their models. The articles are already
public, and this project's index covers only the 38,092-article 2025+ slice,
but note the tension with the dataset licence's no-redistribution condition:
sending text to a third-party API is a disclosure, even of public text. For the
course demo this is accepted and documented. Do not point this at the full
134,667-article corpus, and do not use the free tier if you later obtain
publisher permission on terms that forbid it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://generativelanguage.googleapis.com/v1beta"


def _load_dotenv(path: "os.PathLike | str | None" = None) -> None:
    """Read the project's .env into os.environ, without clobbering the shell.

    Hand-rolled rather than depending on python-dotenv: this is ten lines, the
    venv has no dotenv installed, and adding a dependency for it would be the
    only non-stdlib import in this module. An already-exported variable always
    wins over the file, so `GEMINI_API_KEY=... python rag/generate.py` behaves
    the way anyone would expect.
    """
    env = Path(path) if path else Path(__file__).resolve().parents[1] / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:      # shell beats file
            os.environ[k] = v


_load_dotenv()

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
MAX_PASSAGE_CHARS = 1800        # headline + lede + body opening is where answers are
TIMEOUT = 240       # a single free-tier call was measured at 87s; 90 was too tight

SYSTEM = """You answer questions about Myanmar for a reader who does not read \
Burmese. You are given numbered news passages in Burmese.

Rules, in order of priority:

1. Decide first whether the passages actually answer the question. If they do \
not, set "supported" to false and leave "answer" empty. Passages that are \
merely on a related topic, or about the right kind of event in the wrong place \
or the wrong year, do NOT support an answer. A corpus of Myanmar news will \
often contain something superficially similar to an unrelated question; say so \
rather than stretching.
2. If they do answer it, write the answer in English. Be specific: include \
numbers, dates, place names and named parties exactly as the passages give \
them. Do not add context from your own knowledge — only what the passages say.
3. Cite the passages you used by their numbers, in "citations". Cite only \
passages you actually used. Never cite a number you were not given.
4. If the passages disagree with each other (for example two different \
casualty tallies from different dates), say so in the answer and give both \
with their dates, rather than picking one.
5. Transliterate Burmese place and person names into the spelling an English \
reader would recognise where one exists (မန္တလေး -> Mandalay). Keep the \
Burmese in brackets the first time if the English spelling is uncertain.

Reply with JSON only, no markdown fence:
{"supported": true|false, "answer": "...", "citations": [1,2], \
"unsupported_reason": "..."}

"unsupported_reason" is one short sentence, and only when supported is false."""


def build_prompt(question: str, passages: list[dict]) -> str:
    parts = [f"Question: {question}\n", f"{len(passages)} passages:\n"]
    for n, p in enumerate(passages, 1):
        body = (p.get("text") or p.get("snippet") or "")[:MAX_PASSAGE_CHARS]
        parts.append(
            f"[{n}] {p.get('outlet', '?')}, {p.get('published', '?')}\n"
            f"Title: {p.get('title', '')}\n{body}\n")
    return "\n".join(parts)


def call_gemini(prompt: str, model: str = DEFAULT_MODEL, key: str | None = None,
                retries: int = 3) -> dict:
    key = key or os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set. Get a free key at "
                 "https://aistudio.google.com/apikey and export it.")
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0,
                             "response_mime_type": "application/json"},
    }
    url = f"{API}/models/{model}:generateContent?key={key}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    last = None
    for attempt in range(retries):
        try:
            # Say something BEFORE blocking. A silent retry is indistinguishable
            # from a hang, and the free tier returns 503 often enough that the
            # first run of this looked broken when it was merely waiting.
            print(f"  calling {model}"
                  f"{'' if not attempt else f' (attempt {attempt + 1}/{retries})'}"
                  f"...", file=sys.stderr, flush=True)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                out = json.loads(r.read())
            print(f"  ok in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
            return out
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            last = f"HTTP {e.code}: {detail}"
            # 429 is the free tier's rate limit, 503 its capacity; both are
            # worth waiting out. Print the body every time, not only on the
            # final failure -- an auth or model-name error is in there and
            # hiding it costs more than the noise.
            print(f"  {last}", file=sys.stderr, flush=True)
            if e.code in (429, 500, 503) and attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  retrying in {wait}s", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            break
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
            print(f"  {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            if attempt < retries - 1:
                time.sleep(3)
                continue
    raise RuntimeError(f"Gemini call failed: {last}")


def _text_of(resp: dict) -> str:
    try:
        return resp["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        fb = resp.get("promptFeedback", {})
        raise RuntimeError(f"no candidate in response; promptFeedback={fb}")


def answer(question: str, passages: list[dict], model: str = DEFAULT_MODEL,
           key: str | None = None) -> dict:
    """English answer plus citations, or a refusal, from Burmese passages."""
    if not passages:
        return {"supported": False, "answer": "",
                "unsupported_reason": "no passages retrieved",
                "citations": [], "cited_urls": [], "invalid_citations": []}

    raw = _text_of(call_gemini(build_prompt(question, passages), model, key))
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return {"supported": False, "answer": "", "citations": [],
                "cited_urls": [], "invalid_citations": [],
                "unsupported_reason": "model did not return JSON",
                "raw": raw[:500]}

    # Validate citations against the passages actually supplied. An index
    # outside 1..n is a hallucination and is reported, not silently dropped.
    cites, bad = [], []
    for c in out.get("citations") or []:
        try:
            c = int(c)
        except (TypeError, ValueError):
            bad.append(c)
            continue
        (cites if 1 <= c <= len(passages) else bad).append(c)

    out["citations"] = cites
    out["invalid_citations"] = bad
    out["cited_urls"] = [passages[c - 1].get("url") for c in cites]
    out["model"] = model
    out["n_passages"] = len(passages)
    return out


DEMO = [
    {"outlet": "dvb_burmese", "published": "2025-03-29",
     "title": "နေပြည်တော်တွင် အိမ်ရာများ ပြိုကျ",
     "text": "မတ် ၂၈ ရက်က လွှမ့်သည့် အင်အားပြင်း ငလျင်ဒဏ်ကြောင့် "
             "နေပြည်တော်နှင့် မန္တလေးတွင် အိမ်ရာများ ပြိုကျပြီး "
             "လူ ၁၆၄၄ ဦး သေဆုံးကြောင်း သိရသည်။",
     "url": "https://example.invalid/a1"},
    {"outlet": "bbc_burmese", "published": "2025-03-30",
     "title": "ငလျင်ဒဏ် အကူအညီများ",
     "text": "အိန္ဒိယ၊ တရုတ်နှင့် အာဆီယံနိုင်ငံများမှ ကယ်ဆယ်ရေး အဖွဲ့များ "
             "ရောက်ရှိလာကြောင်း ပြောသည်။",
     "url": "https://example.invalid/a2"},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", help="question to answer")
    ap.add_argument("--passages", type=argparse.FileType(),
                    help="JSON list of passages: [{title,text,url,outlet,published}]")
    ap.add_argument("--demo", action="store_true",
                    help="run two canned passages; no index load, tests the key")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--list-models", action="store_true",
                    help="what this key can call (model names change)")
    args = ap.parse_args()

    if args.list_models:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("GEMINI_API_KEY is not set.")
        with urllib.request.urlopen(f"{API}/models?key={key}", timeout=30) as r:
            for m in json.loads(r.read()).get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    print(f"  {m['name'].removeprefix('models/'):<42}"
                          f"{m.get('displayName', '')}")
        return 0

    if args.demo:
        q = args.q or "How many died in the Myanmar earthquake, and who sent aid?"
        passages = DEMO
    elif args.passages:
        passages = json.load(args.passages)
        q = args.q or sys.exit("--q is required with --passages")
    else:
        ap.error("pass --demo, or --passages with --q")

    out = answer(q, passages, model=args.model)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
