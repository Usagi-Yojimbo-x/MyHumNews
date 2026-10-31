# ui/ — cross-lingual search over the Burmese corpus

Ask a question in English, get ranked Burmese news articles. No translation
anywhere. This is where the project's premise — *reading Myanmar in Burmese
for people who don't read Burmese* — is visible rather than asserted.

## Run

```
.venv/bin/python -m ui.serve
```

Opens <http://127.0.0.1:8770>. `?q=your+question` runs a query on load, so a
search is a shareable link. Startup takes ~15s (a 156MB index, the corpus
bodies, and BGE-M3 onto MPS) and prints what it loaded. Binds 127.0.0.1 only.

```
--port 8770          --host 127.0.0.1      --no-open
--index rag/index    --gazetteer rag/gazetteer/places.json
--data 'data-vm/*/parsed/articles.jsonl'   --page ui/search.html
--questions rag/eval/questions.jsonl
```

`--data` must be the corpus the index was built from, or bodies won't match
the vectors.

## What the numbers mean

```
score = cosine(english_question, burmese_article) + α × gazetteer_hit     α = 0.10
```

- **cosine** — BGE-M3 dense similarity. The English question and the Burmese
  article land in one shared vector space, so this is the cross-lingual step.
- **bonus** — a flat `+α` for articles containing a Burmese place name your
  question named in English. Shown separately per result, so you can see which
  channel earned the rank.

α = 0.10 is the leave-one-out optimum from the 29 known-item questions; the
result is flat from 0.05 to 0.5, so it isn't a knife-edge setting.

Measured effect of the lexical channel (see `rag/eval/RETRIEVAL_RESULTS.md`):

| | Recall@1 | Recall@10 | MRR |
|---|---|---|---|
| dense only | 0.207 | 0.517 | 0.309 |
| dense + gazetteer | 0.276 | 0.586 | 0.389 |

### Why a lexical channel is needed at all

BGE-M3 aligns **translated** names well but **transliterated** ones badly:

```
National Unity Government ↔ အမျိုးသားညီညွတ်ရေးအစိုးရ    0.790   (a translation)
Min Aung Hlaing           ↔ မင်းအောင်လှိုင်              0.468   (a transliteration)
Mandalay                  ↔ မန္တလေး                      0.403
Mandalay                  ↔ နေပြည်တော် (Naypyitaw)       0.440   ← wrong one wins
```

Place names are transliterated, so the dense score can't tell Ramree from
Myingyan. Every known-item miss hinged on one.

## Honest limits, built into the page

- **The index is 38,092 articles published 2025-01 onward** — the frozen
  initial corpus, by an explicit scope decision. The dataset deliverable is
  ~134,667 articles back to Feb 2021, but the RAG is deliberately *not*
  extended to it. Questions about 2021–2024 events will fail, and the page
  says so.
- **Nothing is translated**, by design and because there is no budget for a
  paid API. A non-Burmese reader cannot read the retrieved bodies, so the
  page leans on the language-neutral signals: date, outlet, publisher
  category, the matched place names in English, and the two scores.
- **The 7 HumSet humanitarian labels are not shown** because they do not
  exist for these articles. Only 383 articles are gold-labelled and the
  corpus-wide prediction pass (phase 5) isn't built. Each result card has a
  slot wired to `humanitarian_labels(url)` in `serve.py`, which returns
  nothing on purpose — a plausible wrong label would be worse than a blank.
  Point that function at `predict_corpus.py`'s output when it exists.
- **A query with no gazetteer place name** is reported as such; the page does
  not imply a lexical match happened.

## Implementation notes

**Scoring is copied from `rag/eval/hybrid.py`, not imported.** That file is an
experiment under active change; a UI that breaks mid-ablation is worse than
twelve duplicated lines. `rag/index/`, `rag/gazetteer/` and `rag/eval/` are
read-only inputs here.

**Match forms carry an administrative suffix** (`ဆောမြို့`, never bare `ဆော`).
Burmese has no word boundaries, so short names are substrings of common words:

```
ဆော    alone 117,574 hits   →   ဆောမြို့   418    (Saw township)
ကလေး   alone  18,086        →   ကလေးမြို့ 1,059   (ကလေး also = "child")
ပေါက်  alone  15,915        →   ပေါက်မြို့  924   (ပေါက် also = "to burst")
```

**Text is NFC-normalised before matching.** 6,991 of the 38,092 indexed
articles (18%) are not NFC-normalised — Burmese combining marks can appear in
either order and render identically, so the same word yields different
codepoint sequences and a substring test silently misses. Normalising both
sides recovers 34 gazetteer hits. Note this applies equally to
`rag/eval/hybrid.py`, which does not currently normalise.

**The gazetteer hit matrix is precomputed at startup.** 54 entries × 38,092
articles as a boolean array, built in one pass. A query then ORs a few columns
instead of scanning 38,092 bodies, which is why queries take ~1.5s (almost all
of it BGE-M3 encoding the question) rather than seconds more.

**The left pane's example queries come from `rag/eval/questions.jsonl`**, in
file order, not cherry-picked. K01–K06 includes K05, which the system misses
badly (rank 87). A demo built only from the wins would misrepresent it, and
drawing from the eval set makes the panel a window onto what was actually
measured.

**Design**: palette and layout follow the two references in `ui/inspiration/`
— a near-black field with `#223037` panels and `#DCFB71` as the single loud
colour, in a two-pane split with the composer pinned to the bottom of the
results pane. Lime is reserved for the thing worth looking at: the matched
gazetteer chips and the lexical bonus. A muted lilac `#B9A5E3` carries the
per-place article count and nothing else.

**Burmese on a dark field** needed two fixes that only showed up on render:

- Snippets clamp by `-webkit-line-clamp`, never by `max-height`. A pixel cut
  slices through a row of Burmese and leaves half-drawn diacritics along the
  card edge.
- Weight stays at 600 for titles and 400 for body, never lighter, and
  line-height runs 2.15–2.25 — well above what Latin needs — because stacked
  marks thin out and smear when reversed out of black. Body colour is lifted
  to `#AEBEC6` and the base ink is off-white `#E8EDEF`, since pure white
  haloes against pure black.

**Burmese fonts**: the system stack (`Noto Sans Myanmar`, `Myanmar MN`) comes
first — both ship with macOS, so the page renders offline, as
`classifier/label_web.html` already established. The Google webfont is listed
after them as a fallback for a machine without them. Line-height is 2.0–2.1 on
Burmese text because marks stack above *and* below the baseline and a tight
line clips them.

## Observed on a real run

```
query : Has the military blocked food supply routes in Tanintharyi Region?
matched: Tanintharyi → တနင်္သာရီတိုင်း (733 articles)
rank 1  score 0.6433 = dense 0.5433 + 0.10
        လောင်းလုံးမှာ စစ်ကောင်စီတပ်က ရိက္ခာသယ်ယူတဲ့လမ်းကြောင်း ကန့်သတ်
        rfa_burmese, 2025-01-13  ← this is K01's known target
```

K01's target is rank 1 here (rank 3 dense-only), and K04's target — "How many
people fled the fighting around Myingyan?" — comes back at rank 2 against
rank 110 dense-only. Both match the documented per-question results.
