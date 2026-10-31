#!/usr/bin/env python3
"""Embed the corpus with BGE-M3 into a dense index.

    python -m rag.build_index --data data-vm --out rag/index

Writes:
    <out>/vectors.npy    float32 (n, 1024), L2-normalised
    <out>/meta.jsonl     one line per row: url, outlet, title, published_date, n_chars
    <out>/manifest.json  what was indexed and how

One vector per article from title + the first --max-len tokens. At 512 tokens
that is about 40% of articles complete; a news article's topic is carried by its
headline and lead, and the classifier's head-vs-tail control found no difference
(0.727 vs 0.728). Chunked retrieval is the obvious alternative and is left as an
ablation rather than assumed.

Resumable: re-running skips rows already in meta.jsonl.
"""
from __future__ import annotations

import argparse, glob, json, os, time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "BAAI/bge-m3"
MIN_CHARS = 200          # matches build_dataset.py's usable-body threshold


def iter_corpus(data_dir: str):
    for p in sorted(glob.glob(os.path.join(data_dir, "*", "parsed", "articles.jsonl"))):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data-vm")
    ap.add_argument("--out", default="rag/index")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta_p = os.path.join(args.out, "meta.jsonl")
    vec_p = os.path.join(args.out, "vectors.npy")

    done = set()
    if os.path.exists(meta_p):
        done = {json.loads(l)["url"] for l in open(meta_p, encoding="utf-8") if l.strip()}
        print(f"resuming: {len(done)} rows already indexed")

    rows, seen = [], set()
    for r in iter_corpus(args.data):
        u, body = r.get("url"), (r.get("body") or "")
        if not u or u in seen or u in done or len(body.strip()) < MIN_CHARS:
            continue
        seen.add(u)
        rows.append(r)
        if args.limit and len(rows) >= args.limit:
            break
    print(f"{len(rows)} articles to embed (>= {MIN_CHARS} chars, deduped by url)")
    if not rows:
        print("nothing to do"); return 0

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.fp32 else torch.float16
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=dtype).to(dev).eval()
    print(f"bge-m3 on {dev}, {dtype}, max_len={args.max_len}, batch={args.batch}",
          flush=True)

    old = np.load(vec_p) if os.path.exists(vec_p) else None
    chunks = []
    t0 = time.time()
    with open(meta_p, "a", encoding="utf-8") as mf, torch.no_grad():
        for i in range(0, len(rows), args.batch):
            part = rows[i:i + args.batch]
            texts = [(r.get("title") or "") + "\n" + (r.get("body") or "") for r in part]
            enc = tok(texts, padding=True, truncation=True,
                      max_length=args.max_len, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state[:, 0]
            h = torch.nn.functional.normalize(h, p=2, dim=-1)
            chunks.append(h.float().cpu().numpy())
            for r in part:
                mf.write(json.dumps({
                    "url": r["url"], "outlet": r.get("outlet"),
                    "title": r.get("title"), "published_date": r.get("published_date"),
                    "n_chars": len(r.get("body") or ""),
                }, ensure_ascii=False) + "\n")
            mf.flush(); os.fsync(mf.fileno())
            if (i // args.batch) % 50 == 0:
                n = i + len(part)
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  {n}/{len(rows)}  {rate:.2f} art/s  "
                      f"eta {(len(rows)-n)/max(rate,1e-9)/60:.0f} min", flush=True)
                # checkpoint the vectors so a crash costs minutes, not hours
                np.save(vec_p, np.vstack(([old] if old is not None else []) + chunks))

    V = np.vstack(([old] if old is not None else []) + chunks)
    np.save(vec_p, V)
    json.dump({"model": MODEL, "max_len": args.max_len, "dim": int(V.shape[1]),
               "n": int(V.shape[0]), "dtype": str(dtype), "device": dev,
               "pooling": "cls+l2norm", "min_chars": MIN_CHARS,
               "data": args.data, "built": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"\nwrote {vec_p}  shape {V.shape}  "
          f"{V.nbytes/1e6:.0f} MB  in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
