#!/usr/bin/env python3
"""Chunked variant of the dense index, as a controlled ablation.

    python -m rag.build_index_chunked --data data-vm --out rag/index_chunked

The baseline index embeds title + the first 512 tokens of each article, one
vector per article. An earlier probe compared each known-item target's
whole-article score against its best 700-char chunk and found chunking helps a
lot where the answer sits deep in a long body (K21 +0.176, K22 +0.130,
K17 +0.076) and not at all on the geography misses (~+0.004).

That probe measured the TARGET's score, not its RANK. Max-over-chunks raises
every document's score, distractors included, so a score gain need not be a
rank gain. Only a full chunked index over the same haystack answers that, which
is what this builds.

Same corpus as the baseline (data-vm, 38,092 articles) so the only thing that
differs is pooling. The index is scored by taking each document's best chunk.

Writes:
    <out>/vectors.npy    float32 (n_chunks, 1024), L2-normalised
    <out>/meta.jsonl     one line per CHUNK: url, chunk index, char span
    <out>/manifest.json

Resumable by url: re-running skips articles whose chunks are already written.
"""
from __future__ import annotations

import argparse, glob, json, os, time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "BAAI/bge-m3"
MIN_CHARS = 200          # identical to the baseline index


def iter_corpus(data_dir: str):
    for p in sorted(glob.glob(os.path.join(data_dir, "*", "parsed", "articles.jsonl"))):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                yield json.loads(line)


def chunks_for(title: str, body: str, size: int, stride: int) -> list[str]:
    """Title is prepended to every chunk: it is the strongest topical signal
    and a mid-article chunk without it loses what the article is about."""
    body = body.strip()
    if len(body) <= size:
        return [title + "\n" + body]
    out = []
    i = 0
    while i < len(body):
        piece = body[i:i + size]
        if len(piece) < 120 and out:      # tail too short to stand alone
            break
        out.append(title + "\n" + piece)
        i += stride
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data-vm")
    ap.add_argument("--out", default="rag/index_chunked")
    ap.add_argument("--chunk-size", type=int, default=700)
    ap.add_argument("--stride", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--count-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta_p = os.path.join(args.out, "meta.jsonl")
    vec_p = os.path.join(args.out, "vectors.npy")

    done = set()
    if os.path.exists(meta_p):
        done = {json.loads(l)["url"] for l in open(meta_p, encoding="utf-8") if l.strip()}
        print(f"resuming: {len(done)} articles already chunked")

    rows, seen = [], set()
    for r in iter_corpus(args.data):
        u, body = r.get("url"), (r.get("body") or "")
        if not u or u in seen or u in done or len(body.strip()) < MIN_CHARS:
            continue
        seen.add(u)
        rows.append(r)
        if args.limit and len(rows) >= args.limit:
            break

    plan = []
    for r in rows:
        cs = chunks_for(r.get("title") or "", r.get("body") or "",
                        args.chunk_size, args.stride)
        for k, c in enumerate(cs):
            plan.append((r, k, c))
    print(f"{len(rows)} articles -> {len(plan)} chunks "
          f"({len(plan)/max(len(rows),1):.2f} per article, "
          f"size={args.chunk_size} stride={args.stride})")
    if args.count_only or not plan:
        return 0

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    print(f"bge-m3 on {dev}, fp16, max_len={args.max_len}, batch={args.batch}", flush=True)

    old = np.load(vec_p) if os.path.exists(vec_p) else None
    acc = []
    t0 = time.time()
    with open(meta_p, "a", encoding="utf-8") as mf, torch.no_grad():
        for i in range(0, len(plan), args.batch):
            part = plan[i:i + args.batch]
            enc = tok([c for _, _, c in part], padding=True, truncation=True,
                      max_length=args.max_len, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state[:, 0]
            h = torch.nn.functional.normalize(h, p=2, dim=-1)
            acc.append(h.float().cpu().numpy())
            for r, k, c in part:
                mf.write(json.dumps({"url": r["url"], "chunk": k,
                                     "outlet": r.get("outlet"),
                                     "n_chars": len(c)}, ensure_ascii=False) + "\n")
            mf.flush(); os.fsync(mf.fileno())
            if (i // args.batch) % 100 == 0:
                n = i + len(part)
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  {n}/{len(plan)}  {rate:.1f} chunk/s  "
                      f"eta {(len(plan)-n)/max(rate,1e-9)/60:.0f} min", flush=True)
                np.save(vec_p, np.vstack(([old] if old is not None else []) + acc))

    V = np.vstack(([old] if old is not None else []) + acc)
    np.save(vec_p, V)
    json.dump({"model": MODEL, "max_len": args.max_len, "dim": int(V.shape[1]),
               "n_chunks": int(V.shape[0]), "n_articles": len(rows),
               "chunk_size": args.chunk_size, "stride": args.stride,
               "pooling": "cls+l2norm, max-over-chunks at search",
               "min_chars": MIN_CHARS, "data": args.data,
               "built": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"\nwrote {vec_p} shape {V.shape} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
