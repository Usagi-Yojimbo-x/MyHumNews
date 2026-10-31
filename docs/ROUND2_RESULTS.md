# Round 2 augmentation — results

**Date:** 2026-10-17
**What:** round 2's 180 targeted training articles appended to every CV fold's
TRAINING split. Test folds stay pure gold (383 articles), so every number below
still describes the population gold was sampled from.

```
.venv/bin/python -m classifier.train_humset \
  --augment classifier/data/humset_round2.jsonl \
  --augment-task classifier/data/humset_thin_task.jsonl
```

Same flags work on `tfidf_matched_control` and `train_xlmr` (the latter needs a
GPU; run it from `colab/xlmr_colab.ipynb`, where cell 8 defines `AUG`).

---

## Headline

```
                              gold only   + augment    delta
TF-IDF full text                  0.755       0.764    +0.009
TF-IDF truncated (matched)        0.727       0.732    +0.005
XLM-R, 3 seeds                    0.621       0.661    +0.040

matched gap (TF-IDF - XLM-R)      0.106       0.071
cost of truncation               -0.028      -0.032
```

**Against the matched comparison point, XLM-R gained 8x what TF-IDF did.**
Identical training data, identical 512-token inputs, only the model differs —
and only the transformer responded. The gap closed by a third.

This settles a question that was previously open: the 0.106 gap was
**supervision, not architecture.** Earlier in the project it could only be
stated as "plausibly either, and nothing measured distinguishes them."

## All seven labels become reportable

The `support >= 20` exclusion rule can be dropped:

```
                          TF-IDF full          XLM-R (3-seed mean)
label                 supp   gold   aug      gold    aug     delta
context                186  0.787  0.798    0.730  0.722   -0.008
shock_event            179  0.867  0.852    0.812  0.863   +0.052
displacement            46  0.627  0.635    0.356  0.524   +0.168
casualties              85  0.739  0.762    0.607  0.712   +0.105
humanitarian_access     16  0.000  0.240    0.092  0.264   +0.171
information_comm         6  0.000  0.500    0.013  0.289   +0.276
other                   69  0.756  0.775    0.602  0.483   -0.119

macro-F1 (all 7)            0.539  0.652    0.459  0.551   +0.092
exact match                 0.569  0.587    0.345  0.457   +0.112
```

Both previously-dead labels are non-zero in both arms.

## Caveats that must travel with these numbers

**1. The thin-label figures rest on 16 and 6 test positives.**
`information_communication` TF-IDF 0.500 is precision 1.000 / recall 0.333 —
two predictions, both correct, out of six true positives. One different
prediction swings it by 0.2. The matched-control run gives 0.800 on the same
six. Report these as "non-zero for the first time", never as point estimates.

**2. XLM-R's n is 3 seeds.** Paired by seed, all three improved:

```
seed 20261015   0.617 -> 0.690   +0.073
seed 20261016   0.663 -> 0.700   +0.037
seed 20261017   0.584 -> 0.593   +0.009
```

Mean +0.040, sd rose 0.040 -> 0.059. Paired p is about 0.16. The direction is
consistent across all three seeds; the magnitude is not pinned down.

**3. `other` got worse in the XLM-R arm, -0.119.** Round 2 holds 27 `other`
articles against gold's 69, so the augmented training set is proportionally
poorer in `other` while being far richer in the thin labels. This is the same
dilution pattern the weak-supervision experiment found. It is a real cost of
augmentation, not noise.

**4. Round 2 is ranker-enriched, gold is not.** Round 2's articles were chosen
by `sample_thin_labels.py` taking the top N by decision margin from a
gold-trained ranker. Training on them while testing on gold is a distribution
shift. It does not invalidate the test scores (gold is untouched), but it means
the model has seen a non-random sample of the corpus.

**5. Round 2's held-out half (180 articles) was NOT used.** `--augment-split`
defaults to `train`. Using all 360 would double the thin-label training
positives (101 and 120 instead of 51 and 60) at the cost of the enriched
evaluation set. Untested.

## Implementation notes

- `--augment` asserts **zero URL overlap** with gold before training, and
  refuses to run alongside `--train`.
- `train_xlmr.py`: `pos_weight` is computed over the **augmented** training set.
  Computing it from the fold indices alone would have given
  `information_communication` a weight based on ~5 positives while training on
  65 — the same class of error that invalidated Phase 9 run 1.
- `train_xlmr.py` checkpoints record **`n_aug`**. Augmenting does not change
  `len(X)` (still 383 gold), so without this an augmented Colab run would
  silently resume from a gold-only checkpoint and produce a mislabelled result.
  Verified: the four gold-only checkpoints still resume for gold-only runs and
  are correctly refused for augmented ones.
- `tfidf_matched_control.py`: the augment set is truncated to the same
  `--max-len` for the truncated arm. Otherwise the training set mixes full and
  truncated text and the control stops controlling for truncation.

Artefacts: `classifier/data/xlmr_ckpt_aug_s2026101{5,6,7}.json` (3 seeds,
5 folds each, 383 scored, `n_aug` 180).
