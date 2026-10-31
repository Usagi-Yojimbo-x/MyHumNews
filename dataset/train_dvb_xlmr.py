#!/usr/bin/env python3
"""Fine-tune XLM-RoBERTa on the 19 DVB categories. MULTI-LABEL.

    .venv/bin/python -u -m dataset.train_dvb_xlmr --probe       # token stats, no training
    .venv/bin/python -u -m dataset.train_dvb_xlmr --subset 3000 # pipeline test, ~15 min
    .venv/bin/python -u -m dataset.train_dvb_xlmr               # the real run

The transformer arm of the DVB classifier. The linear arm is dataset/train_dvb.py
+ tune_dvb.py, which tops out at macro-F1 0.658 on the temporal split after
sweeping C, max_features, min_df and per-label thresholds; sweep_dvb.py tests
features and model class. This is the only remaining route that could move the
number substantially rather than by a point.

COMPARABILITY IS THE WHOLE POINT
--------------------------------
Same articles, same 19 labels, same temporal split (train <=2024, test 2025+),
same validation slice carved with the same seed, same threshold search, same
prf(). All of it is imported from tune_dvb rather than reimplemented, so the
macro-F1 printed here is directly comparable to 0.658 and not to a
differently-measured number. Measuring the two arms differently is the usual
way this ablation gets quietly ruined.

ONE HEAD, 19 SIGMOID OUTPUTS
----------------------------
Not 19 separate models. Unlike one-vs-rest TF-IDF, a transformer shares a
representation across labels, which is most of the reason to expect it to help
the weak ones -- and the weak ones are where this task actually loses. The
linear arm's floor is social 0.273 and ethnic 0.246; labels that correlate
(politics/election, peace-war/ethnic, COVID19/health) are predicted blind to
each other by one-vs-rest.

CLASS IMBALANCE, THE LESSON FROM THE HUMANITARIAN ARM
-----------------------------------------------------
classifier/train_xlmr.py's first run used plain BCE while its linear
counterpart used class_weight="balanced". Five of seven labels received zero
predictions and the arm "scored" 0.296; the signal was there and the threshold
was wrong. So, exactly as there:

  pos_weight    BCEWithLogitsLoss(pos_weight = n_neg/n_pos), computed from the
                TRAINING split only -- the direct analogue of scikit-learn's
                class_weight="balanced", which the linear arm has had on all
                along. Clamped at POS_WEIGHT_CAP because the rarest label here
                is ~59 positives in 31,855 and an unclamped weight of 500 makes
                the loss explode rather than balance.

  thresholds    fitted on the validation slice, scored on a test set they never
                saw. Same best_threshold() as the linear arm.

WHAT TO EXPECT, STATED BEFORE RUNNING
-------------------------------------
Unlike the humanitarian arm (306 examples per fold, where overfitting was the
default outcome), 31,855 training examples is a comfortable amount for a 278M
parameter model. If XLM-R cannot beat char n-grams here, that is a real finding
about Burmese in a mostly-non-Burmese sentencepiece vocabulary, not a data-size
excuse.

TRUNCATION IS A CONFOUND, SO IT IS MEASURED
-------------------------------------------
Burmese is not whitespace-segmented and XLM-R's vocabulary was trained mostly on
other languages, so Burmese articles fragment into many subword pieces. --probe
reports the real token distribution and what fraction of each article is lost at
the chosen --max-len. At 256 tokens a long article may be cut to its first few
paragraphs; that is a tradeoff against runtime, not a detail to leave unstated.
Read the probe before trusting the headline number.

RUNTIME, HONESTLY
-----------------
Apple M2, 16 GB, MPS, fp32 (no AMP: GradScaler is CUDA-only and autocast on MPS
is unreliable). MEASURED on this machine at batch 8, not estimated:

    max_len 256  grad_ckpt on    1.89 s/step    2.5 h per epoch
    max_len 512  grad_ckpt on    4.82 s/step    6.2 h per epoch
    max_len 512  grad_ckpt off   6.93 s/step    8.6 h per epoch

Three epochs at 512 is therefore ~19 hours on this laptop. Gradient
checkpointing is kept on because at 512 it is FASTER as well as lighter -- 16 GB
of unified memory cannot hold the full activation graph, so the uncheckpointed
path pays for memory pressure. On a free Colab or Kaggle GPU the same run is
well under two hours, which is the reason --out and the per-epoch config dump
exist: this script detects cuda and runs there unchanged. Every epoch prints a full per-label table and
updates the config file, so a run killed partway still leaves usable results.
Use --subset first to confirm the pipeline before committing the hours.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import dvb_scheme as S                               # noqa: E402
from dataset.tune_dvb import SEED, load, prf, best_threshold      # noqa: E402

MODEL = "xlm-roberta-base"
POS_WEIGHT_CAP = 50.0
OUT = Path("dataset/dvb_xlmr_config.json")


def split(rows):
    """The same temporal split and validation carve-out as tune_dvb."""
    from sklearn.model_selection import train_test_split

    text = [(d.get("title") or "") + "\n" + (d.get("body") or "") for d, _ in rows]
    Y = [set(ls) for _, ls in rows]
    year = [(d.get("published_date") or "")[:4] for d, _ in rows]
    tr_all = [i for i in range(len(text)) if year[i] and year[i] <= "2024"]
    te = [i for i in range(len(text)) if year[i] and year[i] >= "2025"]
    tr, va = train_test_split(tr_all, test_size=0.15, random_state=SEED)
    return text, Y, tr, va, te


def probe(text, idx, max_len):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(idx, size=min(400, len(idx)), replace=False)
    lens = [len(tok(text[i], truncation=False)["input_ids"]) for i in sample]
    lens = np.array(lens)
    chars = np.array([len(text[i]) for i in sample])
    print(f"\n=== tokenizer probe: {MODEL} on {len(sample)} Burmese articles ===")
    print(f"  chars   mean {chars.mean():8.0f}   median {np.median(chars):8.0f}")
    print(f"  tokens  mean {lens.mean():8.0f}   median {np.median(lens):8.0f}"
          f"   p90 {np.percentile(lens, 90):6.0f}   max {lens.max():6.0f}")
    print(f"  tokens per char: {lens.sum() / chars.sum():.3f}"
          f"   (1.0 would mean every character becomes its own piece)")
    for m in (128, 256, 384, 512):
        over = (lens > m).mean()
        kept = np.minimum(lens, m).sum() / lens.sum()
        print(f"  max_len {m:>4}: {over:6.1%} of articles truncated,"
              f" {kept:6.1%} of all tokens kept")
    print(f"\n  at the chosen --max-len {max_len}: "
          f"{(lens > max_len).mean():.1%} truncated, "
          f"{np.minimum(lens, max_len).sum() / lens.sum():.1%} of tokens kept")


def linear_control(text, Y, tr, va, te, labels, max_len):
    """The linear arm, reading only what XLM-R is able to read.

    0.658 was measured on whole articles. XLM-R cannot see a whole article --
    the probe puts the mean at 586 tokens against a 512 ceiling -- so comparing
    XLM-R at 512 tokens against a linear model at full length confounds "which
    model is better" with "which model saw more text". This truncates every
    article to its first max_len XLM-R tokens, decodes it back to text, and runs
    the linear arm on exactly that. The gap between this and 0.658 is the cost of
    truncation alone, charged to neither model.
    """
    from transformers import AutoTokenizer
    from dataset.tune_dvb import evaluate

    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"\n=== linear control: truncating {len(text)} articles to "
          f"{max_len} XLM-R tokens ===", flush=True)
    cut, t0 = [], time.time()
    for a in range(0, len(text), 512):
        chunk = text[a:a + 512]
        ids = tok(chunk, truncation=True, max_length=max_len,
                  add_special_tokens=False)["input_ids"]
        cut.extend(tok.batch_decode(ids, skip_special_tokens=True))
        if a % 10240 == 0 and a:
            print(f"  {a}/{len(text)}  {(time.time()-t0)/60:.1f}m", flush=True)
    kept = sum(len(c) for c in cut) / sum(len(t) for t in text)
    print(f"  done in {(time.time()-t0)/60:.1f}m; {kept:.1%} of characters kept")

    X = lambda idx: [cut[i] for i in idx]
    L = lambda idx: [Y[i] for i in idx]
    macro, rows = evaluate(X(tr), L(tr), X(va), L(va), X(te), L(te),
                           labels, 0.5, 300_000, 2, True, verbose=True)
    print(f"\n" + "=" * 62)
    print(f"  linear arm, WHOLE articles                0.658")
    print(f"  linear arm, first {max_len} XLM-R tokens".ljust(44)
          + f"{macro:.3f}  ({macro - 0.658:+.3f})")
    print("=" * 62)
    print("  The second number is XLM-R's real target.")
    json.dump({"max_len": max_len, "chars_kept": round(kept, 4),
               "macro_f1_truncated": round(macro, 4),
               "macro_f1_full_text": 0.658,
               "f1": {r[0]: round(r[5], 4) for r in rows}},
              open("dataset/dvb_linear_control.json", "w"),
              ensure_ascii=False, indent=1)
    print("  wrote dataset/dvb_linear_control.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset/myhumnews.jsonl"))
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--subset", type=int, default=0,
                    help="train on N sampled articles instead of all; "
                         "for confirming the pipeline runs, not for a result")
    ap.add_argument("--probe", action="store_true",
                    help="tokenizer statistics only, no training")
    ap.add_argument("--linear-control", action="store_true",
                    help="re-score the LINEAR arm on text truncated to --max-len "
                         "XLM-R tokens. This, not 0.658, is the number XLM-R has "
                         "to beat, because 0.658 was measured on whole articles.")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="faster per step, more memory; 16 GB may not hold it")
    ap.add_argument("--no-pos-weight", action="store_true",
                    help="ablation: reproduces the all-negatives collapse")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows = load(args.data)
    labels = sorted(S.F1_TEMPORAL, key=lambda l: -S.F1_TEMPORAL[l])
    text, Y, tr, va, te = split(rows)
    print(f"articles {len(rows)}   train {len(tr)}  val {len(va)}  test {len(te)}")
    print(f"labels {len(labels)}   (temporal split, val carved from train)")
    print("reference: linear arm = 0.658 macro-F1 on this exact split")

    if args.probe:
        probe(text, tr, args.max_len)
        return 0

    if args.linear_control:
        return linear_control(text, Y, tr, va, te, labels, args.max_len)

    if args.subset:
        rng = np.random.default_rng(SEED)
        tr = sorted(rng.choice(tr, size=min(args.subset, len(tr)),
                               replace=False).tolist())
        print(f"\n!! --subset {args.subset}: training on {len(tr)} articles. "
              f"This is a pipeline check; the macro-F1 is NOT comparable to 0.658.")

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    print(f"\ndevice: {dev}   max_len {args.max_len}  batch {args.batch}  "
          f"lr {args.lr}  epochs {args.epochs}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    L = {l: j for j, l in enumerate(labels)}

    def ymat(idx):
        M = np.zeros((len(idx), len(labels)), dtype=np.float32)
        for r, i in enumerate(idx):
            for l in Y[i]:
                if l in L:
                    M[r, L[l]] = 1.0
        return M

    Ytr, Yva, Yte = ymat(tr), ymat(va), ymat(te)

    class DS(Dataset):
        def __init__(self, idx, y=None):
            self.idx, self.y = idx, y

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            return text[self.idx[i]], (None if self.y is None else self.y[i])

    def collate(batch):
        xs = [b[0] for b in batch]
        enc = tok(xs, truncation=True, max_length=args.max_len,
                  padding=True, return_tensors="pt")
        if batch[0][1] is None:
            return enc, None
        return enc, torch.tensor(np.stack([b[1] for b in batch]))

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(labels),
        problem_type="multi_label_classification").to(dev)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()

    # pos_weight = negatives/positives per label, from the TRAINING split only.
    n_pos = Ytr.sum(axis=0)
    pw = np.where(n_pos > 0, (len(tr) - n_pos) / np.maximum(n_pos, 1), 1.0)
    pw = np.minimum(pw, POS_WEIGHT_CAP)
    if args.no_pos_weight:
        pw = np.ones_like(pw)
    print(f"pos_weight: min {pw.min():.1f}  max {pw.max():.1f}"
          f"  (capped at {POS_WEIGHT_CAP})")
    lossf = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pw, dtype=torch.float32, device=dev))

    dl = DataLoader(DS(tr, Ytr), batch_size=args.batch, shuffle=True,
                    collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)

    @torch.no_grad()
    def infer(idx):
        model.eval()
        out = []
        dl2 = DataLoader(DS(idx), batch_size=args.eval_batch, shuffle=False,
                         collate_fn=collate)
        for enc, _ in dl2:
            enc = {k: v.to(dev) for k, v in enc.items()}
            out.append(model(**enc).logits.float().cpu().numpy())
        model.train()
        return np.concatenate(out)

    best = (-1.0, 0)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for step, (enc, yb) in enumerate(dl, 1):
            enc = {k: v.to(dev) for k, v in enc.items()}
            loss = lossf(model(**enc).logits, yb.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run += loss.item()
            if step % 100 == 0:
                el = (time.time() - t0) / 60
                print(f"  ep{ep} step {step}/{len(dl)}  loss {run/step:.4f}"
                      f"  {el:.1f}m  eta {el/step*(len(dl)-step):.0f}m",
                      flush=True)

        # Thresholds from the validation slice, score on test. Never the reverse.
        SV, SB = infer(va), infer(te)
        f1s, rec = [], []
        for l, j in L.items():
            if Ytr[:, j].sum() == 0 or Yte[:, j].sum() == 0:
                continue
            thr = best_threshold(SV[:, j], Yva[:, j].astype(int))
            pred = SB[:, j] >= thr
            gold = Yte[:, j] == 1
            p, r, f = prf(int((pred & gold).sum()), int((pred & ~gold).sum()),
                          int((~pred & gold).sum()))
            f1s.append(f)
            rec.append((l, int(gold.sum()), thr, p, r, f))
        macro = sum(f1s) / len(f1s)
        print(f"\n=== epoch {ep}  macro-F1 {macro:.3f}  "
              f"({macro - 0.658:+.3f} vs linear arm)  "
              f"[{(time.time()-t0)/60:.0f}m] ===")
        print(f"  {'label':<34}{'n':>6}{'thr':>8}{'P':>7}{'R':>7}{'F1':>7}")
        for l, n, t, p, r, f in sorted(rec, key=lambda z: -z[5]):
            print(f"  {l:<34}{n:>6}{t:>8.2f}{p:>7.3f}{r:>7.3f}{f:>7.3f}")

        if macro > best[0]:
            best = (macro, ep)
            json.dump({"model": MODEL, "epoch": ep, "epochs_planned": args.epochs,
                       "max_len": args.max_len, "lr": args.lr,
                       "batch": args.batch, "subset": args.subset or None,
                       "pos_weight_cap": POS_WEIGHT_CAP,
                       "macro_f1_temporal": round(macro, 4),
                       "linear_arm_macro_f1": 0.658,
                       "thresholds": {r[0]: round(r[2], 3) for r in rec},
                       "f1": {r[0]: round(r[5], 4) for r in rec}},
                      open(args.out, "w"), ensure_ascii=False, indent=1)
            print(f"  wrote {args.out}  (best so far)")

    print("\n" + "=" * 62)
    print(f"  linear arm (TF-IDF char_wb + LinearSVC)   0.658")
    print(f"  XLM-R best (epoch {best[1]})".ljust(44) + f"{best[0]:.3f}"
          f"  ({best[0] - 0.658:+.3f})")
    print("=" * 62)
    if args.subset:
        print("  --subset was set: this number is a pipeline check, not a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
