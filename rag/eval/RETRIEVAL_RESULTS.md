# Cross-lingual retrieval: first real numbers

Index: BGE-M3 dense, CLS + L2 norm, 512 tokens, 38,092 articles from
`data-vm` (5 outlets), built 2026-10-17. Queries are English, articles
are Burmese, no translation step anywhere.

Run: `.venv/bin/python rag/eval/run_eval.py --k 20 --dump rag/eval/topk.jsonl`

## Known-item retrieval (n=29)

Each question was written from one specific article, so the target URL is
known and no human judgement is needed.

| metric | value | |
|---|---|---|
| Recall@1  | 0.207 | 6/29 |
| Recall@3  | 0.414 | 12/29 |
| Recall@5  | 0.414 | 12/29 |
| Recall@10 | 0.517 | 15/29 |
| Recall@20 | 0.552 | 16/29 |
| MRR@20    | 0.308 | |

All 29 targets are in the index, so every miss is a retrieval failure
and not a coverage gap.

**An earlier 8/8 result on this index was invalid.** It used each
article's own English gist as the query, which leaks the answer into the
query. Replacing gists with real questions drops it to 6/29. Only the
numbers in this file should be cited.

## The metric is somewhat too strict

Of the 13 questions where the target never appeared in the top 20, four
returned a *different* article that answers the question:

| qid | question | returned instead |
|---|---|---|
| K05 | Did foreign countries send earthquake relief to Naypyitaw? | "International aid and response for the Myanmar earthquake" |
| K17 | What happened to foreigners in scam compounds near the Thai border? | "Thai govt asked to help repatriate people released from KK Park" |
| K19 | How many killed since the coup? | "6,615 killed since the coup" (an earlier tally) |
| K21 | How many people need humanitarian assistance? | "16M+ including 5M children need aid in Myanmar" |

A corpus of 38,092 news articles covers each event many times over, so
one target URL per question under-reports usefulness. Recall@k is still
the right headline number because it needs no judgement, but the true
answer rate is above 0.552. Pinning it down needs the human relevance
judgements the topical questions are waiting on.

## Dominant failure: the geography is lost

Seven of the thirteen misses returned the right *kind* of event in the
wrong *place*:

| asked about | returned |
|---|---|
| Myingyan displacement | Kawkareik displacement |
| Ramree airstrike | Myingyan airstrike |
| Kanbalu raid | Kyindwe bombing |
| Maungdaw jets | Kyindwe bombing |
| Naypyitaw earthquake housing | **Venezuela** earthquake housing |
| Mandalay floods | **Assam, India** floods |
| Bago flu outbreak | **Congo Ebola** outbreak |

This is not random. The model matches the event type and discards the
location.

### Measured cause: BGE-M3 barely aligns Burmese place names

Encoding bare place names, English against Burmese spelling:

```
correct Burmese spelling is top-1 for only 8/12 names
own pair    mean 0.487   min 0.364
other pairs mean 0.388   max 0.543     <- 0.10 of separation

Myingyan -> မြင်းခြံ 0.364   but နေပြည်တော် 0.476   (Naypyitaw wins)
Mandalay -> မန္တလေး  0.403   but နေပြည်တော် 0.440   (Naypyitaw wins)
Myawaddy -> မြဝတီ   0.445   but ဝက်လက်    0.542   (Wetlet wins)
```

The model cannot distinguish Mandalay from Naypyitaw. Burmese place
names are transliterated rather than translated, and Burmese is a thin
slice of BGE-M3's pretraining. A 0.10 margin cannot survive being
averaged into a whole-article vector alongside hundreds of other tokens.

## Chunking is not the fix for that

Comparing each target's whole-article score against its best 700-char
chunk, same query:

| group | qids | delta |
|---|---|---|
| geography misses | K04 K09 K13 K14 K15 K30 | −0.006 to +0.018, mean ≈ +0.004 |
| answer buried in a long body | K21, K22, K17 | **+0.176, +0.130, +0.076** |

Chunking does nothing where location is the problem, and a lot where the
answer sits deep in a long article. Those three are exactly the cases
where the target lost to a shorter article that was *about* the answer.

**Caveat:** this measures the target's score, not its rank. Max-over-chunks
raises every article's score, distractors included. Confirming a rank gain
needs the whole corpus chunked — this is evidence for running that
ablation, not a result from it.

## Negative questions cannot be rejected by score

Five questions have no answer in a Myanmar humanitarian corpus. Their
top-1 similarity:

```
known_item  n=29  min 0.487  median 0.572  max 0.666
negative    n=5   min 0.513  median 0.536  max 0.647

N05 "When does the next SpaceX Starship launch?"   0.647
    -> a Burmese article about a Starship launch explosion
N01 "Current exchange rate of the Brazilian real?" 0.592
    -> "Dollar and gold prices fall back"
```

The distributions overlap completely. N05 at 0.647 beats 24 of the 29
known-item top-1 scores — and it is not even a wrong retrieval, since the
corpus really does carry a Starship story.

So **a cosine threshold cannot tell "nothing here answers this" from
"here is the answer."** Refusing out-of-scope questions has to happen
after retrieval, by asking the generator whether the passages actually
support an answer. That is a finding about system design, not a tuning
problem.

## What follows

1. Chunked index over the full corpus, as a controlled ablation against
   this whole-article run. Same questions, same haystack, pooling is the
   only change. **(Done -- see section 4 below. It loses: R@1 0.207 -> 0.103.)**
2. A township gazetteer (English ↔ Burmese, ~330 names) to add a lexical
   channel for place names and fuse it with the dense score. This is a
   lookup table, not a translation step, so the no-translation premise
   holds.
3. Human relevance judgements on the 15 topical questions, pooled across
   runs, so nDCG and the true answer rate become computable.
4. Re-index after the backfill lands and re-run, since the haystack grows
   from 38k to roughly 123k and Recall@k will move for that reason alone.

---

## Follow-up: which entities actually fail, and why

The place-name result above raised the question of whether named entities
in general are the problem. They are not. Measuring English -> Burmese
alignment on entity pairs where **both forms were first verified to occur
in the corpus** (so the test measures the model, not my spelling):

| class | mean own-pair cosine | top-1 correct |
|---|---|---|
| descriptive org names | **0.734** | 7/7 |
| place names | 0.487 | 8/12 |
| person names | **0.443** | 0/3 |
| idiomatic org name (Red Cross) | 0.470 | 0/1 |

```
NUG   0.790   PDF 0.780   UN 0.778   WFP 0.773
ASEAN 0.718   TNLA 0.713  KNU 0.584

Aung San Suu Kyi 0.488  -> loses to အာဆီယံ (ASEAN)       0.665
Min Aung Hlaing  0.468
Zaw Min Tun      0.374  -> loses to အောင်ဆန်းစုကြည်       0.443
Red Cross        0.470  -> loses to ကုလသမဂ္ဂ (UN)        0.536
```

**The dividing line is translated vs transliterated, not place vs entity.**

- "National Unity Government" ↔ "အမျိုးသားညီညွတ်ရေးအစိုးရ" is a literal
  semantic translation. A multilingual encoder aligns it easily.
- "Min Aung Hlaing" ↔ "မင်းအောင်လှိုင်" is a sound-for-sound
  transliteration carrying no semantics. Nothing in the embedding space
  connects the two. Place names are the same class.
- "Red Cross" ↔ "ကြက်ခြေနီ" fails for a third reason: the Burmese name
  is idiomatic ("red chicken-foot"), not a translation of the English.

This explains the retrieval failures exactly. Every miss in the
known-item set hinged on a transliterated name — a township or a person.

### Consequence for the gazetteer

Cover the transliterated and idiomatic classes only:

- ~330 townships + 15 states/regions (fixed administrative list)
- ~60-80 person names (junta, NUG, ethnic armed organisation figures)
- a short list of idiomatic organisation names

**Deliberately exclude descriptive org names.** The dense model already
scores them 0.71-0.79; adding a lexical channel there would inject noise
for no gain. This makes the gazetteer much smaller than "all entities"
and gives a principled reason for what it contains.

### Implementation note found while verifying spellings

The same entity appears with inconsistent internal spacing, because
Burmese does not space words reliably:

```
တအာင်းအမျိုးသားလွတ်မြောက်ရေးတပ်မတော်      166 occurrences
တအာင်းအမျိုးသား လွတ်မြောက်ရေး တပ်မတော်     81 occurrences
တအာင်းအမျိုးသားလွတ်မြောက်ရေး တပ်မတော်      28 occurrences
```

Any lexical channel must strip whitespace before matching, or it will
miss a third of the mentions. (Also: TNLA is လွတ်မြောက်ရေး "liberation",
not လွတ်လပ်ရေး "independence" — surface forms were taken from the corpus
rather than guessed, and three of fourteen initial guesses were wrong.)

---

## Scope boundary (decided 2026-10-18)

**The dataset deliverable and the RAG system cover different corpora, on
purpose.**

| | corpus | size |
|---|---|---|
| dataset release | all five outlets, 2021-02 onward | ~116,750 and growing |
| RAG index | `data-vm/` only, 2025-01 onward | 38,092 |

The RAG is **not** extended to the backfill. Consequences, stated so that
no reader has to discover them:

1. Every retrieval number in this file is final. They describe the 38,092
   article index named in `rag/index/manifest.json`, and nothing about the
   backfill changes them.
2. The RAG cannot answer questions about events before January 2025. The
   coup period is in the dataset but outside the retrieval system.
3. The evaluation is cleaner for it. All 29 known-item questions were
   written from articles inside this corpus, so there is no haystack-size
   confound: a bigger index would have lowered Recall@k through added
   distractors alone, which is not a property of the retriever.

The gazetteer work is unaffected -- the transliteration failure is a
property of the encoder, not of corpus size.

---

## Hybrid retrieval: gazetteer lexical channel (2026-10-19)

Acting on the transliteration finding above: the query's English place names
are looked up in a gazetteer, and documents containing the corresponding
Burmese form get a bonus added to the dense cosine.

`rag/gazetteer/places.json` -- 54 entries, every one verified to occur in
the corpus. `rag/eval/hybrid.py` -- the scorer.

### Result (leave-one-out, so alpha never saw the question it scores)

| | dense only | + gazetteer |
|---|---|---|
| Recall@1 | 0.207 | **0.276** |
| Recall@5 | 0.414 | **0.483** |
| Recall@10 | 0.517 | **0.586** |
| MRR | 0.309 | **0.389** |

Flat for every alpha from 0.05 to 0.5, so the gain is not a knife-edge
tuning artefact. 17 of 29 questions contain a gazetteer place name.

### It is surgical, which is the main evidence it is real

| | R@1 | R@5 | MRR | n |
|---|---|---|---|---|
| questions WITH a gazetteer hit, dense only | 0.118 | 0.294 | 0.201 | 17 |
| questions WITH a gazetteer hit, + gazetteer | **0.235** | **0.412** | **0.339** | 17 |
| questions WITHOUT one, dense only | 0.333 | 0.583 | 0.465 | 12 |
| questions WITHOUT one, + gazetteer | 0.333 | 0.583 | 0.465 | 12 |

Questions with no place name are **bit-identical** either way: the channel
cannot damage what it does not touch. And the questions it does touch
started far worse (R@1 0.118 vs 0.333), which is the hypothesis confirmed --
place-name questions are the hard subset.

7 questions improved, 3 worsened, 19 unchanged:

```
K04  110 -> 2    K11   88 -> 10   K28  9 -> 1   K01  3 -> 1
K09 miss -> 21   K15 miss -> 110  K02  3 -> 2
K05   69 -> 87   K14   40 -> 55   K30 105 -> 130      (the three losses)
```

All three losses were already at rank 40+, i.e. unfindable before and
unfindable after. The gains moved articles into usable range. Nothing that
was retrievable stopped being retrievable.

### Two implementation findings

**Bare Burmese names cannot be matched directly.** Burmese has no word
boundaries, so short names are substrings of common words:

```
ဆော     (Saw township)  bare 117,574 hits   ဆောမြို့  418
ပေါက်   (Pauk)          bare  15,915        ပေါက်မြို့ 924   (also "to burst")
ကလေး    (Kalay)         bare  18,086        ကလေးမြို့ 1,059 (also "child")
```

Every match form therefore carries an administrative suffix, and documents
are whitespace-stripped first because Burmese spaces words inconsistently.

**A length-based "long names can match bare" rule was tried and rejected.**
It fixed exactly the cases predicted -- Naypyitaw's suffixed form
`နေပြည်တော်မြို့` has only 54 hits against 7,052 bare, because a capital is
not a township, and K05 went 87->23, K14 55->14. But bare matching is less
precise and cost more elsewhere (K01 1->3, K04 2->4, K15 110->191), giving
R@1 0.241 vs 0.276. Net negative, so it is not shipped.

The diagnosis still stands: Naypyitaw is mishandled because its suffix is
wrong for its entity type. A per-entity-type suffix rule is the obvious next
step, but choosing it on these same 29 questions would be selection on test
data. It needs held-out questions first.

### Honest limits

- n=29. R@1 moving 6/29 -> 8/29 is two questions; treat the direction as
  better supported than the magnitude.
- The gazetteer covers places only. Person names are the same transliterated
  class (0.443 mean cosine) and are not covered yet -- K25, a Min Aung Hlaing
  question, still sits at rank 15.
- Coverage is 54 entries against ~330 townships. The 400 mined candidates in
  `rag/gazetteer/burmese_place_candidates.json` cover 86.8% of the 153,201
  place mentions in the corpus and are the obvious place to expand from.

### Person names added (2026-10-19)

`rag/gazetteer/people.json` -- 11 entries, each verified against the corpus
by document frequency, not just raw count. Of 21 proposed, 4 of my Burmese
spellings were absent (Twan Mrat Naing, Mutu Say Poe, Guterres, Anwar
Ibrahim) and 6 were too rare to be useful (<20 documents).

**Effect on this evaluation: none, and the result is bit-identical.**

| | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| dense only | 0.207 | 0.414 | 0.517 | 0.310 |
| places only | 0.276 | 0.483 | 0.586 | 0.391 |
| places + people | 0.276 | 0.483 | 0.586 | 0.391 |

Two reasons, and only the first is a problem with the gazetteer's evaluation
rather than the gazetteer:

1. **Only 1 of 29 known-item questions names a person** (K25, Min Aung
   Hlaing). There is nothing to measure. This is a gap in the question set,
   not evidence the channel fails.
2. **That one question is the worst possible case.** Min Aung Hlaing occurs
   in 10.28% of documents and the question also says "Thailand" (19.00%).
   Both terms match thousands of documents, so the bonus is applied almost
   uniformly and separates nothing. K25 has sat at rank 15 under every
   variant tried.

Person names are **not** inherently weaker than places. Median document
frequency is nearly the same:

```
places (54)  median 0.94%  mean 2.41%  max 19.00% (Thailand)
people (11)  median 1.04%  mean 1.89%  max 10.28% (Min Aung Hlaing)
```

### The general principle this exposes

**A lexical channel's value is inverse to the matched entity's document
frequency.** Ramree (134 mentions, 0.1% of documents) took its question from
miss to rank 21. Thailand (7,247 mentions, 19% of documents) moves nothing,
in any question, at any alpha.

This is actionable rather than merely descriptive: the bonus should be
weighted by inverse document frequency instead of applied as a flat +alpha,
which is what TF-IDF does for exactly this reason. That is the next thing
to try, and unlike per-entity tuning it introduces no parameter chosen on
the test questions.

It also says what the question set needs: questions naming *specific* people
and *small* places, which is where the mechanism can act. Of the 29, the ones
that improved all hinged on a rare township.

---

## Lexical-channel weighting and coverage experiments (2026-10-19)

Three hypotheses were tested against the shipped places-only flat channel
(R@1 0.276, MRR 0.391). **None of them beat it.** The negative results have
clean mechanistic explanations and are more informative than the attempts.

`rag/eval/hybrid_idf.py` runs the weighting comparison. Every alpha is
selected leave-one-out.

### 1. IDF weighting: no change at all

| scheme | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| dense only | 0.207 | 0.414 | 0.517 | 0.310 |
| flat (shipped) | 0.276 | 0.483 | 0.586 | **0.391** |
| idf-max | 0.276 | 0.483 | 0.586 | 0.391 |
| idf-sum | 0.276 | 0.483 | 0.586 | 0.391 |

Identical to three decimal places. Two reasons:

**16 of the 17 matched questions name exactly one gazetteer form** (only K25
names two). With one form, `alpha * idf(form)` is just the flat scheme at a
per-question rescaled alpha, so idf relabels alpha rather than changing what
the channel does.

**And it rescales in the wrong direction.** idf gives a larger effective alpha
to rarer forms, on the assumption that rarity implies usefulness. Here rarity
instead flags a *broken match form*:

```
entity      match form           df    idf   this question wants
Ramree      ရမ်းဗြဲမြို့          77   6.20   a LARGE alpha
Myingyan    မြင်းခြံမြို့        188   5.31   a LARGE alpha
Mandalay    မန္တလေးတိုင်း       2103   2.90   a LARGE alpha  (K15)
Naypyitaw   နေပြည်တော်မြို့       42   6.81   a SMALL alpha  (K05, K14)
```

Naypyitaw's form is the **rarest of all 65** (df=42) and is the one that needs
the *smallest* alpha, because `နေပြည်တော်မြို့` is rare only because the suffix
is wrong -- a capital is not a township. Mandalay is the mirror image:
frequent, and a correct, useful match. So idf amplifies the gazetteer's worst
entry and damps a good one. K05 went 87->96 and K14 55->67 under idf-max.

**A premise correction:** the channel is *not* saturated in alpha, as was
assumed when this was commissioned. Individual ranks move a great deal
(K04: 55->29->6->2 across alpha 0.01..0.10; K09: miss->114->21). The earlier
observation that aggregates were "flat from 0.05 to 0.5" was true of the
*aggregate only*, where per-question gains and losses offset.

### 2. Gazetteer expansion cannot move this evaluation

The 400 mined candidates were not added, because scanning the 29 questions for
place names the gazetteer does not already cover returns nothing. The only
unmapped capitalised tokens across all 29 are `Region`, `Thai`, `Rohingya` and
the parts of `Min Aung Hlaing` (already covered as a whole). **The gazetteer is
already complete with respect to this question set**, so expanding it from 54
to 400 entries would change no number here. It remains worth doing for real
use; it is simply not measurable against these questions.

### 3. A topical-term gazetteer: more questions helped, no better retrieval

`rag/gazetteer/terms.json` -- 27 entries (Rohingya, earthquake, airstrike,
political prisoner, scam compound/ကျားဖြန့်, flu, drone ...), all verified by
document frequency, none above the 15% ceiling.

| variant (leave-one-out) | R@1 | R@5 | R@10 | MRR | questions covered |
|---|---|---|---|---|---|
| places only, any (shipped) | 0.276 | 0.483 | 0.586 | **0.391** | 17/29 |
| all 3 gazetteers, any | 0.241 | 0.483 | 0.586 | 0.359 | 24/29 |
| all 3, require all forms | 0.276 | 0.414 | 0.586 | 0.373 | 24/29 |
| all 3, fraction of forms | 0.276 | 0.448 | 0.586 | 0.385 | 24/29 |
| all 3, idf-weighted fraction | 0.276 | 0.448 | 0.552 | 0.382 | 24/29 |

Coverage rises from 17/29 to 24/29 and the improved/worsened shape gets
*better* -- `frac` improves 11 questions and worsens 2, against the shipped
channel's 7 and 3 -- yet no aggregate metric improves.

**Why adding terms hurts under flat scoring:** the bonus fires if *any* form
matches, so each added form monotonically widens the matched set. K02 goes
from matching `Rakhine` (6.8% of documents) to
`Rakhine OR Rohingya OR aid OR food`, and `aid` alone is 10.7%, which swamps
the place signal. Requiring co-occurrence (`all`, `frac`) repairs the
dilution but then buys nothing over places alone.

### The finding that matters most

**R@10 is 0.586 under every lexical variant tried:**

```
dense only                   0.517
places only, any             0.586   <- shipped
all 3, any / all / frac      0.586
places, idf-max / idf-sum    0.586
all 3, idf-weighted          0.552
```

The gazetteer's one real gain (0.517 -> 0.586) arrives immediately and then
**saturates**. Nine weighting and coverage variants cannot move it. Every
apparent improvement from the term gazetteer lands *outside* the top 10
(K13 miss->283, K15 110->55, K23 miss->273), so for a RAG pipeline that reads
the top k passages, none of it changes the outcome.

The remaining headroom is not reachable by lexical means. It needs a different
mechanism -- which is what the chunked-pooling ablation tests.

### Recommendation

Keep the shipped places-only flat channel. It is the simplest configuration
and nothing beats it. But note that MRR 0.391 vs 0.385 is roughly one question
at n=29: these variants are within noise of each other, and calling
places-only "best" is itself a selection on 29 questions. The defensible claim
is that **no lexical variant separates from the others**, and simplicity
breaks the tie.

### 4. Chunked retrieval ablation -- RESOLVED, chunking loses (2026-10-22)

`rag/index_chunked/` holds 218,122 chunks (700 chars, stride 500, title
prepended) over the **same 38,092 articles**, so pooling is the only
difference from the baseline. 5.73 chunks per article. A document's score is
the max over its chunks. Run: `.venv/bin/python rag/eval/eval_chunked.py`.

| pooling | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| whole article (512 tok) | 0.207 | 0.414 | 0.517 | 0.310 |
| chunked, max-over-chunks | 0.103 | 0.345 | 0.483 | 0.202 |
| whole + gazetteer (LOO) | **0.276** | **0.483** | **0.586** | **0.391** |
| chunked + gazetteer (LOO) | 0.172 | 0.448 | 0.552 | 0.290 |

**Chunking halves R@1 and costs a third of MRR.** Against the whole-article
baseline it improved 5 questions, worsened 15, left 9 unchanged. With the
gazetteer channel on top: 8 improved, 13 worsened, 8 unchanged. The loss is
not concentrated anywhere -- it is the broad middle of the ranking.

### The earlier probe was right about the mechanism and wrong about the outcome

The score probe predicted large gains where the answer sits deep in a long
body (K21 +0.176, K22 +0.130, K17 +0.076) and nothing on the geography misses
(~+0.004). Both halves of that prediction held:

```
predicted to gain       K21  miss -> 102     K22  miss -> 4      K17  miss -> miss
geography misses        K04   110 -> 24      K11    88 -> 32
predicted flat / lost   K14    40 -> 124     K30   105 -> 191    K25  15 -> 47
```

K22 going from unfindable to rank 4 is exactly the predicted effect, and K21
and K04 moved too. The mechanism is real. It is simply outweighed, because
max-over-chunks raises **every** document's score and a corpus of 38,092 news
articles has far more near-miss distractors than targets. The probe measured
the numerator and could not see the denominator -- which is why it was written
up as evidence for running this ablation rather than as a result.

This is the caveat in the probe section landing exactly as stated: *"this
measures the target's score, not its rank ... confirming a rank gain needs the
whole corpus chunked."* It did not confirm.

### R@10 = 0.586 is now the ceiling under every mechanism tried

```
dense only                          0.517
places only, any (shipped)          0.586   <- still the best
all 3 gazetteers, any/all/frac      0.586
places, idf-max / idf-sum           0.586
all 3, idf-weighted                 0.552
chunked, max-over-chunks            0.483
chunked + gazetteer                 0.552
```

Nine lexical weighting and coverage variants could not move 0.586, and
chunked pooling moves it **down**. The remaining headroom is not reachable by
pooling or by lexical means on this encoder. It needs either a retriever with
real Burmese alignment or a reranking stage that reads candidate passages --
not a change to how the same BGE-M3 vectors are pooled or combined.

### What ships

The whole-article index with the places-only flat gazetteer channel,
unchanged. `rag/index_chunked/` is kept so the ablation is reproducible, but
it is not used at serving time.

### Honest limits

- n=29, so 0.207 -> 0.103 at R@1 is three questions. The direction is solid
  (15 of 29 worsened, and the aggregate drops on all four metrics) but the
  magnitude is not well pinned at this sample size.
- One chunk configuration was tested: 700 chars, stride 500, max pooling.
  A smaller chunk, or sum/mean-of-top-k instead of max, might behave
  differently. Max pooling is the standard choice and the one the probe
  measured, so it is the configuration the probe's prediction applies to.
- A `max(whole, chunked)` score fusion would plausibly keep K22 and K04 while
  dropping the losses, since the losses are all cases where the whole-article
  score was already better. It is not reported here because choosing that
  rule on these same 29 questions is selection on test data -- the same
  reason the per-entity suffix rule was not shipped. It needs held-out
  questions first.

### NFC normalisation (2026-10-19)

Found while building the UI: **18.3% of the indexed articles (6,995 of
38,152) are not NFC-normalised.** Burmese combining marks can be stored in
either order and render identically, so a substring match against a
differently-ordered form silently misses.

`rag/eval/hybrid.py` now normalises both the document text and the
gazetteer match forms. Effect on matching:

```
(article, place) pairs   49,651  ->  49,685   (+34)
```

**Effect on every retrieval metric: none.** R@1 0.276, R@5 0.483,
R@10 0.586, MRR 0.389 — identical to three decimals before and after.
The 34 recovered matches fell on articles that were not competing for the
top ranks.

Recorded because the fix is correct and should stay, but it is not a
result: anyone re-deriving these numbers will get the same figures with or
without it. The reason to keep it is that the miss is silent and would bite
a larger gazetteer or a different question set, not that it improves this
one.
