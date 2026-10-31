#!/usr/bin/env python3
"""BGE-M3 dense embeddings for cross-lingual retrieval.

BGE-M3 is XLM-R's vocabulary with 8192 positions and retrieval-contrastive
pretraining. Dense retrieval = CLS token of the last hidden state, L2
normalised. No sentence-transformers dependency needed.

The point of this model in this project: an English question and a Burmese
article that answers it land near each other in the same vector space, so
retrieval needs no translation step.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "BAAI/bge-m3"
_tok = None
_model = None


def load(device: str | None = None):
    global _tok, _model
    if _model is None:
        dev = device or ("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _model = AutoModel.from_pretrained(MODEL).to(dev).eval()
        _model._dev = dev
        print(f"bge-m3 loaded on {dev}", flush=True)
    return _tok, _model


@torch.no_grad()
def encode(texts: list[str], max_len: int = 1024, batch: int = 8,
           progress: bool = False) -> np.ndarray:
    tok, model = load()
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(model._dev)
        h = model(**enc).last_hidden_state[:, 0]      # CLS
        h = torch.nn.functional.normalize(h, p=2, dim=-1)
        out.append(h.float().cpu().numpy())
        if progress and (i // batch) % 20 == 0:
            print(f"  {min(i+batch, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(out)


def search(qvec: np.ndarray, index: np.ndarray, k: int = 10):
    """Both are L2-normalised, so a dot product IS cosine similarity."""
    sims = index @ qvec.T            # (n_docs, n_queries)
    order = np.argsort(-sims, axis=0)[:k]
    return order, np.take_along_axis(sims, order, axis=0)
