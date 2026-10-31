---
language: [my]
license: other
license_name: mixed-cc-by-4.0-annotations-publisher-copyright-text
license_link: LICENSE
task_categories: [text-classification, text-retrieval]
tags: [burmese, myanmar, news, humanitarian, low-resource]
size_categories: [100K<n<1M]
extra_gated_heading: "Request access to MyHumNews"
extra_gated_description: >-
  This corpus contains the full text of news articles that remain the
  copyright of their publishers, four of which are exile or banned Myanmar
  outlets whose reporting names people who may be at risk. Access is granted
  for non-commercial research only.
extra_gated_prompt: >-
  By requesting access you confirm that you have read the Licensing and
  Access section of the dataset card, and that you accept the three
  conditions below. Access may be revoked, and the dataset withdrawn, at the
  request of any of the five publishers.
extra_gated_fields:
  Full name: text
  Institution or organisation: text
  Institutional email: text
  Intended use (one or two sentences): text
  I will use this dataset for non-commercial research only: checkbox
  I will not attempt to identify, locate or contact individuals named in these articles, and will not redistribute the article text: checkbox
  I understand the article text is the publishers' copyright and is not covered by the open licence on the annotations: checkbox
extra_gated_button_content: "Request access"
---
# MyHumNews: a Burmese news corpus (2021–2026)

134,667 Burmese-language news articles from five Myanmar outlets, February 2021
(the military coup) to September 2026, with 20 topic labels.

> **Gated release.** Full article text is included. Access is granted on request
> for non-commercial research — see [Licensing and access](#licensing-and-access).

| | |
|---|---|
| articles | 134,667 |
| date range | 2021-02-01 .. 2026-09-26 |
| body text | 372.7M characters, 83.0% Burmese script |
| body length | median 1,744 chars |
| outlets | 5 |
| labels | 20 classes |
| publisher-assigned (`topic_dvb19`) | 80,235 (59.6%) |
| model-predicted (`topic_dvb19_predicted`) | 44,075 (32.7%) |

| outlet | articles | publisher-assigned | model-predicted |
|---|---|---|---|
| `dvb_burmese` | 78,083 | 62,018 (79.4%) | 12,640 (16.2%) |
| `rfa_burmese` | 23,534 | **0 (0.0%)** | 19,974 (84.9%) |
| `irrawaddy_burmese` | 14,570 | 9,931 (68.2%) | 2,932 (20.1%) |
| `bbc_burmese` | 9,279 | 8,109 (87.4%) | 1,072 (11.6%) |
| `myanmar_now_burmese` | 9,201 | **177 (1.9%)** | 7,457 (81.0%) |

## Fields

| field | type | fill | notes |
|---|---|---|---|
| `url` | string | 100% | canonical publisher URL; the dedupe key |
| `outlet` | string | 100% | one of the five slugs above |
| `title` | string | 100% | |
| `body` | string | 100% | extracted article text; 45 rows empty |
| `published_date` | string | 100% | ISO-8601 UTC |
| `modified_date` | string | 96.3% | |
| `category` / `categories` | string / list | 99.8% | the **publisher's own** categories, unaltered |
| `tags` | list[string] | 16.2% | RFA only, in English |
| `topic_dvb19` | list[string] | 59.6% | **publisher-assigned** labels |
| `topic_dvb19_predicted` | list[string] | 32.7% | **model-predicted**, only where the above is empty |
| `topic_dvb19_margin` | list[float] | 32.7% | classifier margin, aligned with the column above |
| `author` | string | 81.1% | often an organisation, not a person |
| `language` | string | 100% | always `my`; the corpus is Burmese throughout |
| `cms_locale` | string | 100% | the publisher CMS's locale, **not** the article language |
| `post_id`, `content_sha256`, `scraped_at` | string | 100% | |
| `section`, `is_prose` | string, bool | 17.7%, 10.8% | Irrawaddy and Myanmar Now only |

Irrawaddy and Myanmar Now report `cms_locale: en-US` while their bodies are
81–86% Burmese script. `language` states the measured truth; the CMS value is
kept separately because it is real provenance.

## Labels

The 20 labels are **real editorial desks from real newsrooms** — 19 of DVB's,
plus `လူ့အခွင့်အရေး` (human rights) from Irrawaddy and BBC. None is this
project's invention. `dataset/dvb_scheme.py` and `dataset/outlet_crosswalk.py`
derive them from the shipped `categories` field, so the mapping is reproducible.

| label | n | | label | n |
|---|---|---|---|---|
| `ငြိမ်းချမ်းရေး/စစ်ရေး` peace/military | 22,210 | | `COVID19` | 1,934 |
| `နိုင်ငံတကာ` international | 18,747 | | `လူ့အခွင့်အရေး` human rights | 1,554 |
| `နိုင်ငံရေး` politics | 16,872 | | `ရွေးကောက်ပွဲ` election | 1,375 |
| `crime` | 6,602 | | `ပညာရေး/နည်းပညာ` education | 1,308 |
| `sport` | 5,524 | | `အနုပညာ` arts | 987 |
| `စီးပွားရေး` economy | 5,080 | | `အမျိုးသမီး/လူငယ်` women/youth | 781 |
| `သဘာဝပတ်ဝန်းကျင်` environment | 3,101 | | `စိုက်ပျိုးရေး` agriculture | 574 |
| `လူမှုရေး` social | 2,599 | | `သတင်းမီဒီယာ` news media | 352 |
| `တိုင်းရင်းသား` ethnic | 2,393 | | `ဘာသာရေး` religion | 268 |
| `အလုပ်သမား` labour/migrant | 2,031 | | `ကျန်းမာရေး` health | 1,999 |

### Predicted labels: read this before using them

`topic_dvb19_predicted` is in a **separate column on purpose**. Ignore it and
you get publisher labels only. Opting in is explicit:

```python
labels = row["topic_dvb19"] or row["topic_dvb19_predicted"]
```

TF-IDF char_wb(3,5) + LinearSVC, trained on DVB only, thresholds set for
precision 0.80 on held-out DVB. **Cross-outlet it mostly fails.** Measured on
18,217 non-DVB articles that carry publisher labels:

```
economy 0.86 · arts 0.85 · health 0.76 · peace/military 0.72 · COVID19 0.69
environment 0.67 · international 0.60 · crime 0.60 · election 0.58
politics 0.55 · education 0.52 · labour 0.47 · sport 0.12
```

Only 2 of 13 testable labels reach 0.80. The cause is **prevalence shift** —
thresholds calibrated on DVB's label frequencies do not transfer to another
newsroom's. The predicted column is most trustworthy on DVB (12,640 articles,
no transfer involved); 19,974 of the predictions are RFA, where no validation
is possible at all because RFA has no publisher labels.

**Five labels are never predicted.** Four because the publisher applies them
inconsistently — near-duplicate articles agree on `social` only 15% of the
time, `ethnic` 18%, `women/youth` 25%, `religion` 29%, against `sport`'s 99%.
The fifth, `human_rights`, has too little DVB data to learn from. `COVID19` is
also suppressed for 2025+ articles (11 positives in that period vs 1,237 in
training).

## Known limitations

- **Four labels are applied inconsistently by the publisher**, and this is
  measured: near-duplicate agreement predicts achievable F1 at r = +0.874
  across all labels. `social`, `ethnic`, `women/youth` and `religion` bound
  macro-F1 at roughly 0.73 for any model. A low score on them is a property of
  the annotation, not of your model.
- **`COVID19` is unmeasurable on a 2025-onward test set** — 11 positives there
  against 1,282 in 2021–2024. Exclude it or state the prevalence.
- **RFA and Myanmar Now have essentially no publisher labels** (0.0% and 1.9%);
  both file almost everything under a generic "news" category. If label
  provenance matters, exclude those two outlets.
- **10,357 articles (7.7%) have no label at all**, deliberately left empty
  rather than given a catch-all class. They are ordinary articles (median body
  1,674 chars) that the publisher filed under format categories only and the
  classifier declined at its precision target.
- **45 articles have an empty `body`** — genuine photo/video posts from the coup
  weeks. Filter on `len(body) > 0` for text tasks.
- **61 BBC URLs are permanently gone** (404 from bbc.com), so they are absent.
- **Outlet selection is not neutral.** All five are independent or exile
  outlets. State media is absent, so this corpus does not represent Myanmar's
  media landscape and should not be used to measure it.
- **The humanitarian gold labels are NOT in this release** and were sampled from
  the 2025-onward subset only, so they are not representative of a corpus that
  is two-thirds pre-2025.

## Licensing and access

**Gated.** Access on request for non-commercial research. The gate exists for
two reasons, and only the first is copyright.

- **Annotations, label schemes, metadata and code**: CC-BY-4.0. The authors' own
  contribution.
- **Article text** (`title`, `body`): copyright of BBC, DVB, The Irrawaddy,
  Myanmar Now and RFA. Included for non-commercial research use. An open licence
  **cannot** cover it, which is why the frontmatter says `license: other`.
- **Takedown**: zwehtetaung.doh@gmail.com. Any of the five publishers may
  request removal, with no reason required, and it will be honoured. Individuals
  named in the articles may use the same address.

Four of the five outlets are exile or banned Myanmar media reporting on a
hostile military government. Their articles name local sources, witnesses and
families — people who are not public figures and may be at real risk. Those
articles are already public one at a time; a bulk machine-readable archive is a
different object. Gating does not prevent misuse, but it means access is
requested rather than scraped, there is a record of who holds the data, and
users have accepted a condition against re-identification.

Users accept: non-commercial research only; no attempt to identify, locate or
contact individuals named in the articles, and no redistribution of the text.

## Reproducing

```
.venv/bin/python dataset/build_release.py      # publisher labels
.venv/bin/python dataset/predict_topics.py     # reads the release, predicts
.venv/bin/python dataset/build_release.py      # again, attaching predictions
```

Three steps, because step 2 reads step 1's output. The build unions four crawl
trees, dedupes by `url` (7,607 duplicates across 142,274 source rows),
normalises inconsistent fields, and asserts the expected per-outlet counts —
refusing to write if they do not match. `predict_topics.py --dry-run` reports
the threshold and transfer measurements without writing.

Raw HTML was written to disk and `fsync`ed before parsing, so the corpus is
reproducible from the archive layer. Each record carries `content_sha256` over
its extracted text.

## Citation

```bibtex
@misc{myhumnews2026,
  title  = {MyHumNews: A Burmese News Corpus for Humanitarian
            Classification and Cross-Lingual Retrieval},
  author = {Zwe Htet Aung},
  year   = {2026},
  note   = {Sejong University, BDAPP},
  url    = {https://huggingface.co/datasets/usagi-yojimbo/MyHumNews}
}
```
