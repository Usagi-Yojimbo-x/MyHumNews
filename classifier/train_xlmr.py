#!/usr/bin/env python3
"""
Fine-tune XLM-RoBERTa for the 7 HumSet labels. MULTI-LABEL.

    python -m classifier.train_xlmr --probe        # tokenizer stats only, no training
    python -m classifier.train_xlmr               # 5-fold CV on the gold set
    python -m classifier.train_xlmr --epochs 6 --lr 1e-5

The XLM-R arm of the proposal's TF-IDF-vs-XLM-R ablation. The TF-IDF arm is
`train_humset.py` at macro-F1* 0.755 / exact 0.569 (5-fold CV, 383 articles).

HELD TO THE SAME PROTOCOL AS THE TF-IDF ARM, DELIBERATELY
---------------------------------------------------------
Same 383 articles, same 5 folds from the same seed, same label-set-signature
grouping, same `other`-exclusivity repair, same metrics, same MIN_SUPPORT=20
rule for which labels count. If the two arms were measured differently the
comparison would be worthless, which is the usual way this ablation goes wrong.

One head with 7 sigmoid outputs and BCE loss, not 7 separate models: unlike
one-vs-rest TF-IDF, a transformer can share a representation across labels,
which is most of the reason to expect it to do better on the rare ones.

WHAT TO EXPECT, STATED BEFORE RUNNING
-------------------------------------
XLM-R base has 278M parameters and will be fine-tuned on 306 training examples
per fold. That is a ratio where overfitting is the default outcome, and it is
entirely possible this arm LOSES to a linear model on character n-grams. That
would be a real finding, not a failed experiment: it is the honest answer to
"does a multilingual transformer help at the data scale this project actually
has", and the proposal asks for the comparison, not for the transformer to win.

CLASS IMBALANCE IS HANDLED, AND THE FIRST RUN PROVES WHY IT MUST BE
-------------------------------------------------------------------
The first version of this script used plain BCE with no positive-class
weighting while the TF-IDF arm used `class_weight="balanced"`. The result was a
collapse: 5 of 7 labels received ZERO predictions and the arm "scored" 0.296.

Inspecting the dumped logits showed the signal was present and the threshold was
wrong -- `casualties` separated positives from negatives by +0.77 logits, but
every one of its logits was negative, so nothing crossed 0. Re-thresholding
alone lifted macro-F1* from 0.296 to 0.640.

Two corrections, both of which make this comparable to the TF-IDF arm:

  pos_weight    BCEWithLogitsLoss(pos_weight = n_neg/n_pos), computed PER FOLD
                from the training split only. This is the direct analogue of
                scikit-learn's class_weight="balanced".

  thresholds    chosen per label on the TRAINING fold, never on the test fold.
                Tuning on test would have given 0.640, which is not a number
                anyone may report.

Two things are measured rather than assumed:

  TRUNCATION   Burmese is not whitespace-segmented and XLM-R's sentencepiece
               vocabulary was trained mostly on other languages, so Burmese
               fragments into many pieces. At 512 tokens a 2,321-character mean
               article may be cut substantially. `--probe` reports the real
               token distribution and the fraction lost, because a 40%
               truncation rate would be a confound in the comparison, not a
               detail.

  THE SEEDS    a single fine-tuning run on 306 examples is noisy. The TF-IDF arm
               needed a paired-across-splits test to distinguish a 0.015 effect
               from 0.055 fold noise; this arm is noisier still. `--seeds N`
               repeats the whole CV and reports the spread, so the comparison
               against 0.755 is made against a distribution rather than a point.
"""

from __future__ import annotations

import argparse
import collections
import json
import gc
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

from classifier import humset, train_humset as T

MODEL = "xlm-roberta-base"
GOLD = Path("classifier/data/humset_gold.jsonl")
TASK = Path("classifier/data/humset_task.jsonl")


def load():
    X, Y, rows = T.build(GOLD, TASK)
    return X, Y, rows


def probe(X, max_len: int) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    lens = [len(tok(x, truncation=False)["input_ids"]) for x in X]
    lens.sort()
    n = len(lens)
    print(f"\n=== tokenizer probe: {MODEL} on {n} Burmese articles ===")
    print(f"  chars   mean {sum(len(x) for x in X) / n:,.0f}")
    print(f"  tokens  mean {st.mean(lens):,.0f}  median {st.median(lens):,.0f}  "
          f"p90 {lens[int(.9 * n)]:,}  max {lens[-1]:,}")
    print(f"  tokens per char: {st.mean(lens) / (sum(len(x) for x in X) / n):.2f}")
    over = sum(1 for l in lens if l > max_len)
    lost = sum(max(0, l - max_len) for l in lens)
    print(f"\n  at max_length={max_len}:")
    print(f"    articles truncated : {over}/{n} ({over / n:.1%})")
    print(f"    tokens discarded   : {lost:,} of {sum(lens):,} ({lost / sum(lens):.1%})")
    if over / n > 0.25:
        print(f"\n  WARNING: truncation this heavy is a CONFOUND in the TF-IDF")
        print(f"  comparison -- the TF-IDF arm sees the whole article. Report it,")
        print(f"  and consider --max-len 512 vs longer as its own ablation.")


def run_cv(X, Y, rows, max_len, epochs, lr, batch, seed, quiet=False,
           ckpt: Path | None = None, grad_ckpt: bool = True,
           amp: bool = True,
           Xa: list[str] | None = None,
           Ya: list[set[str]] | None = None):
    """5-fold CV. If `ckpt` is given, each fold's scores are written there as
    soon as it finishes and completed folds are skipped on a re-run.

    This exists because the first two runs held every fold's scores in memory
    and wrote nothing until fold 5, so a kill, a flat battery or a Colab
    disconnect threw away the whole job. Each fold is ~25 min; losing five of
    them to a dropped session is avoidable.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.model_selection import StratifiedKFold

    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    print(f"  device: {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    L = humset.LABELS
    idx = {l: i for i, l in enumerate(L)}

    class DS(Dataset):
        def __init__(self, xs, ys):
            self.xs, self.ys = xs, ys
        def __len__(self):
            return len(self.xs)
        def __getitem__(self, i):
            # NO padding here -- the collator pads to each batch's own longest
            # item. `padding="max_length"` made every batch 512 wide regardless
            # of content and was a large part of the 15 GB peak footprint.
            e = tok(self.xs[i], truncation=True, max_length=max_len)
            item = {k: torch.tensor(v) for k, v in e.items()}
            y = torch.zeros(len(L))
            if self.ys is not None:
                for lab in self.ys[i]:
                    y[idx[lab]] = 1.0
            item["labels"] = y
            return item

    def collate(batch):
        """Pad to the longest item in THIS batch, not to max_len."""
        keys = [k for k in batch[0] if k != "labels"]
        n = max(len(b["input_ids"]) for b in batch)
        out = {}
        for k in keys:
            pad = tok.pad_token_id if k == "input_ids" else 0
            out[k] = torch.stack([
                torch.cat([b[k], torch.full((n - len(b[k]),), pad,
                                            dtype=b[k].dtype)])
                for b in batch])
        out["labels"] = torch.stack([b["labels"] for b in batch])
        return out

    sig = ["|".join(sorted(y)) for y in Y]
    c = collections.Counter(sig)
    sig = [s if c[s] >= 5 else "RARE" for s in sig]

    # Extra TRAINING-only articles. They never enter a test fold, so `scores`
    # and every reported metric still cover exactly the gold set.
    Xa = Xa or []
    Ya = Ya or []

    scores = [None] * len(X)
    done_folds: set[int] = set()
    if ckpt and ckpt.exists():
        saved = json.loads(ckpt.read_text())
        if (saved.get("seed") == seed and saved.get("n") == len(X)
                and saved.get("n_aug", 0) == len(Xa)):
            for k_s, per in saved["folds"].items():
                for i_s, sc in per.items():
                    scores[int(i_s)] = sc
                done_folds.add(int(k_s))
            print(f"  resuming: folds {sorted(done_folds)} already done",
                  flush=True)
        else:
            print(f"  checkpoint is for a different seed/size/augment -- ignoring",
                  flush=True)

    def save(k, tes):
        if not ckpt:
            return
        cur = {"seed": seed, "n": len(X), "n_aug": len(Xa), "folds": {}}
        if ckpt.exists():
            old = json.loads(ckpt.read_text())
            if (old.get("seed") == seed and old.get("n") == len(X)
                    and old.get("n_aug", 0) == len(Xa)):
                cur = old
        cur["folds"][str(k)] = {str(i): scores[i] for i in tes}
        tmp = ckpt.with_suffix(".tmp")
        tmp.write_text(json.dumps(cur))
        tmp.replace(ckpt)

    skf = StratifiedKFold(5, shuffle=True, random_state=T.SEED)
    for k, (tr, te) in enumerate(skf.split(X, sig), 1):
        if k in done_folds:
            continue
        torch.manual_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=len(L), problem_type="multi_label_classification"
        ).to(dev)
        # pos_weight = negatives/positives per label, from the TRAINING split
        # only. Mirrors class_weight="balanced" in the TF-IDF arm. Capped so a
        # 6-positive label does not get a weight of 50 and destabilise training.
        Ytr = [Y[i] for i in tr] + Ya
        ntr = len(Ytr)
        pw = []
        for lab in L:
            npos = sum(1 for y in Ytr if lab in y)
            pw.append(min((ntr - npos) / max(npos, 1), 20.0))
        pos_weight = torch.tensor(pw, device=dev)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        dl = DataLoader(DS([X[i] for i in tr] + Xa, Ytr),
                        batch_size=batch, shuffle=True, collate_fn=collate)
        # Trades ~30% more compute for a large cut in activation memory.
        # Worth it on a 17 GB laptop; usually not on a 16 GB T4, where VRAM is
        # dedicated rather than shared with the OS -- hence the flag.
        if grad_ckpt:
            model.gradient_checkpointing_enable()
        total = epochs * len(dl)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=total, pct_start=0.1)
        use_amp = amp and dev == "cuda"
        # bf16 where the GPU supports it (A100/L4), else fp16 with a scaler.
        amp_dt = (torch.bfloat16
                  if use_amp and torch.cuda.is_bf16_supported()
                  else torch.float16)
        scaler = torch.amp.GradScaler("cuda",
                                      enabled=use_amp and amp_dt == torch.float16)
        model.train()
        step = 0
        for ep in range(epochs):
            ep_loss = []
            for b in dl:
                b = {kk: v.to(dev) for kk, v in b.items()}
                y = b.pop("labels")
                if use_amp:
                    with torch.autocast("cuda", dtype=amp_dt):
                        out = model(**b)
                    loss = lossf(out.logits.float(), y)
                    scaler.scale(loss).backward()
                else:
                    out = model(**b)
                    loss = lossf(out.logits, y)
                    loss.backward()
                ep_loss.append(loss.item())
                if use_amp:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if not quiet:
                # EPOCH MEAN, not the last batch. The first version printed the
                # last batch, which is noise and cannot show overfitting.
                print(f"    fold {k} epoch {ep + 1}/{epochs} "
                      f"loss {sum(ep_loss) / len(ep_loss):.4f}", flush=True)
        model.eval()

        def infer(ids):
            dl2 = DataLoader(DS([X[i] for i in ids], None),
                             batch_size=batch, collate_fn=collate)
            out = []
            with torch.no_grad():
                for b in dl2:
                    b = {kk: v.to(dev) for kk, v in b.items() if kk != "labels"}
                    out.append(model(**b).logits.float().cpu().numpy())
            return np.concatenate(out)

        # Thresholds from the TRAINING fold. Never from the test fold.
        trl = infer(tr)
        thr = {}
        for l in L:
            col = trl[:, idx[l]]
            y = np.array([1 if l in Y[i] else 0 for i in tr])
            cands = np.unique(np.round(col, 3))
            best, bt = -1.0, 0.0
            for t in cands:
                p = col > t
                tp = int((p & (y == 1)).sum())
                fp = int((p & (y == 0)).sum())
                fn = int(((~p) & (y == 1)).sum())
                f = T.prf(tp, fp, fn)[2]
                if f > best:
                    best, bt = f, float(t)
            thr[l] = bt
        if not quiet:
            print(f"    fold {k} thresholds "
                  + " ".join(f"{l[:4]}={thr[l]:+.2f}" for l in L), flush=True)

        logits = infer(te)
        for j, i in enumerate(te):
            # store threshold-shifted scores so downstream `> 0` stays correct
            scores[i] = {l: float(logits[j][idx[l]]) - thr[l] for l in L}
        save(k, te)
        del model, opt, sched, lossf, trl, logits
        gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()
        elif dev == "cuda":
            torch.cuda.empty_cache()
        if not quiet:
            print(f"  fold {k} done", flush=True)
    return scores


def report(Y, scores, rows, title):
    raw = [{l for l in humset.LABELS if s[l] > 0} for s in scores]
    rep = [T.repair(s, r) for s, r in zip(scores, raw)]
    n_empty = sum(1 for r in raw if not r)
    m = T.report(Y, rep, rows, title)
    print(f"\n  empty raw outputs before repair: {n_empty}/{len(raw)} "
          f"({n_empty / len(raw):.1%})")
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--augment", type=Path,
                    help="extra LABELLED articles appended to every CV fold's "
                         "TRAINING split only; test folds stay pure gold")
    ap.add_argument("--augment-task", type=Path,
                    help="text source for --augment (it carries no bodies)")
    ap.add_argument("--augment-split", default="train",
                    help="keep only --augment rows with this `split` value; "
                         "empty string uses all of them")
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--ckpt", type=Path,
                    default=Path("classifier/data/xlmr_ckpt.json"),
                    help="per-fold checkpoint; re-running resumes from it")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing (faster; use on a GPU "
                         "with dedicated VRAM, e.g. Colab/Kaggle T4)")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable mixed precision (CUDA only)")
    args = ap.parse_args()

    X, Y, rows = load()

    Xa: list[str] = []
    Ya: list[set[str]] = []
    if args.augment:
        Xa, Ya, arows = T.build(args.augment, args.augment_task or args.augment,
                                split=args.augment_split or None)
        overlap = {r["url"] for r in arows} & {r["url"] for r in rows}
        if overlap:
            sys.exit(f"--augment leaks into the test set: {len(overlap)} shared "
                     f"urls, e.g. {sorted(overlap)[:3]}")
        print(f"augment +{len(Xa)} articles into every training fold "
              f"(split={args.augment_split or 'all'}, 0 overlap with gold)")
        for lab in humset.LABELS:
            g = sum(1 for y in Y if lab in y)
            a = sum(1 for y in Ya if lab in y)
            print(f"    {lab:<28} gold {g:>4}  +augment {a:>4}")
    print(f"gold {len(X)} articles, {len(humset.LABELS)} labels, "
          f"{sum(len(y) for y in Y) / len(Y):.2f} labels/article")

    if args.probe:
        probe(X, args.max_len)
        return 0

    probe(X, args.max_len)
    print(f"\nXLM-R: max_len={args.max_len} epochs={args.epochs} lr={args.lr} "
          f"batch={args.batch} seeds={args.seeds}")

    macros = []
    for s in range(args.seeds):
        seed = T.SEED + s
        print(f"\n--- seed {seed} ---", flush=True)
        ck = None if args.no_ckpt else (
            args.ckpt if args.seeds == 1
            else args.ckpt.with_name(f"{args.ckpt.stem}_s{seed}.json"))
        scores = run_cv(X, Y, rows, args.max_len, args.epochs, args.lr,
                        args.batch, seed, quiet=(args.seeds > 1), ckpt=ck,
                        grad_ckpt=not args.no_grad_ckpt, amp=not args.no_amp,
                        Xa=Xa, Ya=Ya)
        m = report(Y, scores, rows, f"XLM-R, seed {seed}")
        macros.append(m)
        if args.dump and s == 0:
            raw = [{l for l in humset.LABELS if sc[l] > 0} for sc in scores]
            rep = [T.repair(sc, r) for sc, r in zip(scores, raw)]
            with open(args.dump, "w") as fh:
                for r, y, p, sc in zip(rows, Y, rep, scores):
                    fh.write(json.dumps({
                        "url": r["url"], "site": r["site"],
                        "labels": sorted(y), "pred": sorted(p),
                        "logits": {k: round(v, 4) for k, v in sc.items()},
                    }, ensure_ascii=False) + "\n")
            print(f"\nwrote {args.dump}")

    print(f"\n{'=' * 72}")
    print(f"  XLM-R  macro-F1* over {len(macros)} seed(s): "
          + "  ".join(f"{m:.3f}" for m in macros))
    if len(macros) > 1:
        print(f"         mean {st.mean(macros):.3f}  sd {st.stdev(macros):.3f}")
    # 0.727, NOT 0.755. XLM-R reads only the first 512 tokens, so the matched
    # comparison is TF-IDF on the same truncated text (tfidf_matched_control.py).
    # 0.755 is TF-IDF on FULL text and is the wrong baseline for this arm.
    base = float(os.environ.get("TFIDF_BASELINE", "0.727"))
    print(f"  TF-IDF macro-F1* {base:.3f} (matched-input: same 512-token truncation)")
    print(f"         (TF-IDF on FULL text is 0.755 -- not the comparison point)")
    d = st.mean(macros) - base
    print(f"  difference {d:+.3f}  ->  "
          f"{'XLM-R wins' if d > 0.02 else 'TF-IDF wins' if d < -0.02 else 'indistinguishable at this n'}")
    print(f"{'=' * 72}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
