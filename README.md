# MyHumNews — Burmese news corpus

Crawlers for building a Burmese-language news corpus across multiple outlets.
All five outlets are implemented: **DVB Burmese**, **BBC Burmese**, **RFA
Burmese**, **The Irrawaddy** and **Myanmar Now**.

## Architecture

One shared engine, one small adapter per outlet:

```
crawler/core.py         SQLite queue, fetcher, retry/backoff, raw+JSONL, logging
crawler/sites/dvb.py    discover() + parse()      203 lines
crawler/sites/bbc.py    discover() + parse()      264 lines
crawler/sites/rfa.py    discover() + parse()      279 lines
crawler/sites/irrawaddy.py  discover() + parse()  218 lines
crawler/sites/myanmarnow.py discover() + parse()  186 lines
run.py                  CLI: python run.py <site> <phase>
```

An adapter supplies exactly two things — how to find article URLs, and how to
turn one page into a record. Everything else is shared, so the retry logic and
the JSONL schema are identical across outlets by construction. Adding an outlet
is one file.

`dvb_crawler.py` is the original standalone version, superseded by
`crawler/sites/dvb.py`. Both produce byte-identical output (verified against 20
archived pages); it is kept only because a running VM crawl was started with it.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install requests beautifulsoup4 lxml
```

## Use

```bash
# test slice
.venv/bin/python run.py bbc run --limit 20

# full crawl, Jan 2025 onward
.venv/bin/python run.py dvb run          # ~12h
.venv/bin/python run.py bbc run          # ~2h
.venv/bin/python run.py rfa run          # ~1.6h
.venv/bin/python run.py irrawaddy run    # ~3h  (needs `curl` installed)
.venv/bin/python run.py myanmarnow run   # ~1.4h

# phases separately
.venv/bin/python run.py dvb discover
.venv/bin/python run.py dvb fetch

.venv/bin/python run.py dvb status
.venv/bin/python run.py dvb fetch --retry-failed
```

Interrupt with Ctrl-C or `kill -9` at any time; rerunning the same command
resumes and never re-fetches a completed URL.

Flags: `--start`, `--end`, `--delay`, `--out`, `--limit`, `--max-pages`,
`--retry-failed`, `--quiet`. Date range and output root default from `run.py`;
politeness delay defaults per-site from the adapter.

## Output

```
data/<site>/
  raw/<id>.html.gz             gzipped source HTML, re-parseable offline
  parsed/articles.jsonl        one JSON object per line
  crawl.db                     SQLite queue (urls, discovery_state)
  crawl.log                    one line per request outcome
```

JSONL fields: `url`, `outlet`, `title`, `body`, `published_date`, `category`,
`scraped_at`, `content_sha256`, plus `post_id`, `categories[]`, `modified_date`,
`author`, `language`.

`content_sha256` is the SHA-256 of the UTF-8 body text (for dedup across outlets
that syndicate the same story).

Re-parse without re-crawling:

```python
import gzip
from crawler.sites import dvb, bbc

rec = dvb.parse(gzip.open("data/dvb/raw/941928.html.gz","rb").read(),
                "https://burmese.dvb.no/post/941928")
```

## Burmese text

Stored **exactly as served**: no Zawgyi/Unicode conversion, no whitespace
collapsing, no NFC normalization. `&nbsp;` decodes to `\xa0` and is kept.

DVB's pre-~2019 archive is Zawgyi-encoded; content from Jan 2025 onward is clean
Unicode. If the corpus stays inside the 2025+ range, the conversion step may be a
no-op — validate before converting.

## DVB Burmese (recon, 2026-09-23)

Next.js + Strapi behind Cloudflare; server-rendered, so no JS/Selenium needed.

- `robots.txt`: `Allow: /`, `Disallow: /api/`, `Crawl-delay: 2`. The Strapi REST
  API under `/api/` is disallowed and is not used. `news-please` is explicitly
  blocked, so identify honestly and stay at >=2s.
- **No RSS/Atom.** `/rss` and `/feed` soft-200 to the homepage.
- `sitemap.xml` -> 21 shards; shards 1-14 hold 138,950 `/post/<id>` URLs
  (complete archive, 2010-present). Shard 0 has the category list.
- Sitemap `<lastmod>` is a bulk-migration timestamp, **not** the publish date.
- **Post IDs are not chronological** (941928 = today, 999460 = June 2026), so
  there is no binary-search-by-ID shortcut.
- **Category listings are strictly date-descending** — this is the only cheap
  route to a date-bounded crawl, and is what `discover` walks.
- `/categories/news` pages 1-698 cover Jan 2025 -> today (~13,960 articles);
  ~16-20k distinct once other categories are unioned in (measured overlap 17.4%).

### Parsing gotchas

- **`<h1>` is sidebar chrome** ("About Us" / "Follow Us"), never the headline.
  Title comes from JSON-LD `headline`.
- **JSON-LD `articleSection` holds only the primary category.** `div.categories`
  has the full list. Both are stored (`category` and `categories[]`).
- Body is `div.full_content`; inner nesting differs between site eras, so the
  parser takes the container rather than pinning a selector chain.

### Known coverage limit

Discovery walks category listings, so an article with **no category assigned**
would appear in the sitemap but in no listing and would be missed. To quantify:

```bash
# compare discovered IDs against the sitemap's 138,950
```

If the gap is material, add a top-up pass over sitemap IDs not present in `urls`.

---

## BBC Burmese (recon, 2026-09-24)

Stack: Next.js pages router. All article data is in the `__NEXT_DATA__` JSON
blob, so parsing is JSON, not HTML scraping.

`robots.txt`: `/burmese/` is allowed (49 Disallow rules cover search and other
sections). **No Crawl-delay published** — we use 2s anyway.

### Discovery is the hard part

Every obvious route is a dead end:

- `/burmese/sitemap.xml` is **stale** — 100 URLs, all September 2014.
- RSS (`feeds.bbci.co.uk/burmese/rss.xml`) is a **rolling 20-item window**.
  Fine for incremental updates, useless for backfill.
- Topic listings **hard-cap at 8 pages** (~185 articles) and always show the
  most recent items, so busy topics cannot reach back to Jan 2025.

So the site's own navigation cannot cover the date range. We use the **Wayback
CDX index as a URL index only**, then fetch every article live from bbc.com so
stored content is current. Topic pages are walked afterwards for anything too
recent to be archived.

Two CDX traps, both handled in `crawler/sites/bbc.py`:

1. **`from=`/`to=` filter on capture date, not publish date.** They happily
   return a 2023 article archived in 2025. We request all captures and filter
   client-side on the *earliest* one. This is safe: a page cannot be archived
   before it is published, so `first_capture >= published` always, and the
   filter can never drop an in-range article.
2. **The `articles*` wildcard matches junk.** Of 17,139 raw ids, ~11,500 were
   fragments like `000c` or `0049e072`, all of which 404 on live bbc.com.
   Anchoring on the real CPS id shape (`c` + 10–14 alphanumerics) is what keeps
   the queue clean.

Result: 6,629 valid article URLs, **2,919 first-captured on/after 2025-01-01**.

### Parsing — every assumption inverts from DVB

| | DVB | BBC |
|---|---|---|
| Structured data | JSON-LD `NewsArticle` | none — `__NEXT_DATA__` |
| Title | JSON-LD only (`<h1>` is chrome) | `<h1>` is the real title |
| Dates | ISO-8601 string | **epoch milliseconds** |
| Discovery | category pages, unlimited depth | Wayback CDX + capped topics |

Body is built from `content.model.blocks` where `type == "text"`; images,
promos, related links and disclaimers are excluded. Title comes from
`promo.headlines.headline`, falling back to `seoHeadline` (often the only one
populated). Categories come from `metadata.topics[].topicName`.

### Known coverage limit

Unlike DVB — where the sitemap proves 138,950 articles exist and category
pagination reaches all of them — **BBC coverage is partial**. We can only see
articles the Wayback Machine happened to archive, plus the most recent ~185 per
topic. 2,919 archived URLs over ~21 months is well below BBC Burmese's likely
publication volume, so treat the BBC subset as a sample, not a census, and say
so in any writeup. Worth measuring properly before drawing cross-outlet
conclusions about volume.

---

## RFA Burmese (recon, 2026-09-25)

Stack: **Arc XP (Fusion)**. Every article embeds a `Fusion.globalContent` JSON
object with the full article model — body blocks, taxonomy, credits, dates — so
parsing is JSON. A JSON-LD `NewsArticle` block is also present and is used as a
fallback.

`robots.txt`: articles allowed. Only `/burmese/search/?query=*` and any path
ending `story_archive` are disallowed — the latter would have been a convenient
archive route, so we deliberately do not touch it. No Crawl-delay; we use 2s.

### Discovery — the cleanest of any outlet

Arc exposes **date-addressable daily sitemaps**:

```
/arc/outboundfeeds/burmese/sitemap/<YYYY-MM-DD>/?outputType=xml
```

The published index lists only the last 365 days, but arbitrary earlier dates
resolve fine. So discovery is just "enumerate every day in range" — one request
per day, ~633 for Jan 2025 onward. Exact, complete, no pagination guessing, no
over-fetching of out-of-range articles, no external archive needed.

`sitemap/`, `sitemap2/` and `sitemap3/` are aliases returning identical content.

### Publishing collapse — the headline finding

RFA Burmese output declines steeply through 2025 and **stops entirely for seven
months**. Sampled articles/day (4 days per month):

```
2025-01  17.2  #################
2025-02  13.5  ##############
2025-03   8.8  #########
2025-04   5.8  ######
2025-05   2.0  ##
2025-06   0.0            <-- goes dark
  ...              (Jun-Dec 2025: zero)
2025-12   0.0
2026-01   2.0  ##        <-- partial resumption
2026-02   0.5
2026-03   1.2  #
2026-04   5.8  ######
2026-05   5.0  #####
```

Those zero months appear in RFA's own 365-day sitemap index and return a valid
but empty urlset, so **the gap is what RFA published, not a hole in our crawl**.
Treat it as a real signal, not missing data. The timing coincides with RFA's
2025 funding crisis; worth confirming against a source before asserting cause.

Estimated corpus: **~2,300 articles** for Jan 2025 onward. Roughly 1.6h to crawl
(21 min discovery + ~77 min fetch).

### Parsing

| field | source |
|---|---|
| title | `headlines.basic` |
| body | `content_elements[]` where `type == "text"` |
| published_date | `first_publish_date` (ISO-8601, already correct format) |
| category | `taxonomy.primary_section.name` (Burmese, e.g. သတင်းများ) |
| categories | `taxonomy.sections[].name` |
| author | `credits.by[0].name` (Burmese bylines) |
| tags | `taxonomy.tags[].text` |

`Fusion.globalContent` cannot be extracted with a regex — the object contains
braces inside string literals — so the adapter brace-matches while tracking
string state. `<h1 class="b-headline">` is the real title here, and RFA article
URLs embed the publish date as `MMDDYYYYHHMMSS` before `.html`.

---

## The Irrawaddy, Burmese (recon, 2026-09-25)

Stack: **WordPress** (JNews theme) with Yoast sitemaps, behind **Cloudflare**.

`robots.txt` is `User-agent: *` with **no Disallow lines at all** — crawling is
explicitly permitted, no Crawl-delay. We use 2s regardless.

### Cloudflare — the only outlet requests cannot fetch

python-requests is challenged on **every** path (403 + a "Just a moment..."
interstitial). curl with a full browser header set succeeds. Identical requests
seconds apart:

```
curl      -> 200
requests  -> 403 (challenge)
```

Same headers, same URL. The discriminator is the **TLS/HTTP2 handshake
fingerprint**, which Cloudflare scores alongside headers. Both signals matter —
dropping `Sec-Fetch-*` from the curl request is enough to get challenged too.

So this adapter sets `transport = "curl"` and `core.CurlFetcher` handles it.
Worth being precise about what this is: **we are not spoofing a browser
fingerprint.** curl presents its own honest fingerprint, Cloudflare allows it,
and we use it on a site whose robots.txt invites crawling. The challenge is
blanket CDN protection, not a policy signal aimed at crawlers.

Measured reliability once warmed up: 10/10 on articles, 12/12 across sitemaps,
category pages and articles. The interstitial arrives as a 403 carrying HTML
rather than a network error, so `CurlFetcher` treats it as retryable — it
appears to be IP warm-up behaviour and clears on a later attempt.

**This makes `curl` a runtime dependency for this outlet only.** The other three
still use requests.

### Discovery

`sitemap_index.xml` lists 226 sub-sitemaps, of which **68 are
`post-sitemap*.xml`** (1,000 URLs each, ~68k articles). They are *not* ordered
by publish date — post-sitemap68 spans 2014 to 2026 — so there is no early
stop. We fetch all 68 (cheap) and filter on the date embedded in the URL.

Because the date is in the URL, **nothing out of range is ever downloaded** —
the only outlet besides RFA where that holds.

### The section trap

Articles live under **many** section prefixes, one or two segments deep:

```
/news/2026/07/11/414709.html
/video/2026/07/12/414722.html
/article/2026/09/19/416361.html
/opinion/viewpoint/2026/07/18/414874.html      <- two segments
/news/short-news/2026/09/05/416064.html        <- two segments
/cartoon/  /lifestyle/  /business/  /feature/  /election/  /laborrights/
```

An `/article/`-only pattern matches **12%** of the site (694 URLs instead of
4,405). The regex must stay general; the section is captured and stored.

### Corpus: 4,405 articles, Jan 2025 → Sep 2026

Even coverage, ~200/month, **no gaps** — the healthiest of the four outlets.

```
  1928  video               417  opinion/viewpoint      15  business
  1042  news                106  news/short-news        12  opinion/editorial
   694  article             103  cartoon                 3  laborrights
                             73  opinion/interview       2  feature
```

`video` is 44% of the corpus. Those pages do carry real body text (~1,200
chars), so they are kept, but each record stores `section` and `is_prose` so
video/cartoon can be excluded from text analysis with one filter.

Fetch time ≈ 3h at 2s.

### Parsing

| field | source |
|---|---|
| title | JSON-LD `headline` (fallback `h1.jeg_post_title`) |
| body | `div.entry-content` (figures stripped) |
| published_date | JSON-LD `datePublished` |
| categories | JSON-LD `articleSection` — **already a list**, mixes English and Burmese |
| section, is_prose | derived from the URL path |

JSON-LD sits inside an `@graph` wrapper and the type is `Article`, not
`NewsArticle`. `inLanguage` reports `en-US` even though the content is Burmese —
recorded but **not trustworthy**.

---

## Myanmar Now, Burmese (recon, 2026-09-25)

Stack: **WordPress** (Yoast), no CDN challenge. Plain `requests` works.

Burmese edition lives at `/mm/` (`/my/` and `/burmese/` are 404). `robots.txt`
is a Yoast block with `Disallow:` — empty, so everything is allowed — and no
Crawl-delay. We use 2s.

### The open WordPress REST API makes this the cleanest of the five

```
/mm/wp-json/wp/v2/posts?after=<iso>&before=<iso>&per_page=100&page=N
```

Native date filtering, native pagination, and an `X-WP-Total` header giving the
exact corpus size before crawling. **25 requests enumerate the whole range in
~50 seconds**, and discovery returned exactly 2,455 URLs — matching
`X-WP-Total` precisely, which is a verifiable completeness check no other
outlet offers.

It also sidesteps the trap that bit The Irrawaddy: the API returns each post's
canonical `link`, so there is **no URL pattern to guess** and no way for a
too-narrow regex to silently drop most of the site.

### Corpus: 2,455 articles, Jan 2025 → Sep 2026

Even coverage, ~110–160/month, **no gaps**. All under a single `/mm/news/<id>/`
section. September 2026 is low (61) only because the month is incomplete.

Fetch time ≈ 1.4h at 2s.

### Bodies come from HTML, not the API — deliberately

`content.rendered` in the API carries the article body, so the entire corpus
could be pulled in ~25 requests instead of ~2,455. We fetch the HTML pages
anyway, to keep the raw layer consistent with the other four outlets (gzipped
HTML, re-parseable offline) and to keep a single parsing path.

If a much faster re-run is ever wanted, switching `discover()` to emit records
straight from the API is a small change — at the cost of a JSON raw layer that
differs from the rest of the corpus.

### Parsing

Matches The Irrawaddy closely: JSON-LD `Article` inside an `@graph`, body in
`div.entry-content`, `articleSection` as a list. `h1.entry-title` is the
fallback title. As on The Irrawaddy, `inLanguage` claims `en-US` on Burmese
content and should not be trusted.

---

# Terminology: "outlet label", not "gold"

Three label sources appear throughout, and they are **not** equally reliable.
They were originally all called "gold", which was wrong and actively misleading;
`source` values and field names now say what each one is.

| `source` | n | what it is | agreement with a human |
|---|---:|---|---|
| `hand` | 194 | a person read the article and decided | by definition, the reference |
| `outlet` | 29,206 | the outlet's own section name, mapped by `taxonomy.py` | 64.5%, or 79.8% excluding the foreign desk |
| `predicted` | 8,739 | the model's output | 0.60–0.73 |

An **outlet label** is scraped page metadata, not verified truth. Nobody read
the article:

    article      DEMU အမြန်ရထားကို ခဲဖြင့်ပေါက်သူ ၃ ဦး ကို ဖမ်းဆီးအရေးယူ
    scraped      categories = ['မြန်မာသတင်း', 'မှုခင်း']
                 မြန်မာသတင်း  -> DROPPED (generic "news")
                 မှုခင်း      -> crime
    outlet label crime

The correct term for this is **distant supervision**: existing metadata used as
a cheap, noisy proxy for labels. "Gold" conventionally means human-verified
ground truth, which only the 194 `hand` rows are.

The naming mattered. Calling 29,206 weak labels "gold" made them sound
authoritative and is part of why the circularity in Phase 5 went unnoticed for
so long — a figure measured against something called "gold" reads as accuracy,
whereas the same figure measured against "the outlet's filing convention"
obviously does not.

In `heldout_pred.jsonl` the field is `outlet_label`, not `outlet`, because
`outlet` already means the publication (`"outlet": "dvb_burmese"`) in the corpus
schema.

None of this diminishes the outlet labels' value: they are the only labels
available at corpus scale, they are what the model trained on, and at 79.8%
agreement on non-foreign topics they are a serviceable training signal. The
error was reporting agreement with them as though it were accuracy.

# Phase 2 — Topic classifier

## Zawgyi: not needed

Checked all 38,152 articles: **zero Zawgyi**. The conversion stage planned at
the start of the project is a no-op and was removed. Running a converter over
this corpus would have corrupted it.

## Why the outlets' own labels can't be the target

416 distinct labels across five outlets, and only **16 appear in more than one
outlet**. Granularity is wildly inconsistent — BBC has 323 fine-grained tags,
RFA has 8 of which one (`သတင်းများ`, "News") covers 96% of its articles.

They are also a mix of five different things:

| kind | examples | usable |
|---|---|---|
| real topics | `ငြိမ်းချမ်းရေး/စစ်ရေး`, `စီးပွားရေး` | yes |
| format/placement | `Lead Story`, `Video`, `ဆောင်းပါး`, `Cartoon` | no |
| generic "news" | `သတင်း`, `သတင်းများ`, `မြန်မာသတင်း` | no |
| geography | `အမေရိကန်`, `တရုတ်`, `ရန်ကုန်` | foreign only |
| named entities | `မင်းအောင်လှိုင်`, `ဒေါ်နယ်ထရမ့်` | via their domain |

`classifier/taxonomy.py` maps them onto **10 topic classes**. Coverage of the
75,192 label assignments: 53.5% mapped, 43.1% deliberately dropped as
non-topical, 3.4% unmapped long tail.

**This mapping needs domain review.** It was built by reading Burmese label
strings; every downstream metric inherits its errors, and a mis-mapped class is
invisible in the accuracy number.

### The combining-mark trap

The first version of the mapping silently missed ~1,000 label assignments.
Burmese combining marks can appear in either order and render identically:

```
corpus:  ... U+103A U+1037      (asat, then dot-below)
typed:   ... U+1037 U+103A      (dot-below, then asat)
```

Different byte strings, so dict lookups miss. Both sides are now NFC-normalised
— in the label mapping and in article text before feature extraction. The raw
layer and the crawled JSONL are left exactly as collected; normalisation
happens downstream only.

## Dataset

`python -m classifier.build_dataset` → 29,203 labelled articles from 38,152.

```
read      38152
dup_url       5      restart artefacts
too_short    55      <200 chars
dup_body     83      RFA programme-announcement boilerplate
no_topic   8806      format/placement labels only
kept      29203      76.5%
```

Split is **temporal** by default. A random split was expected to leak (five
outlets covering one event) but measured 0.847/0.758 against temporal's
0.846/0.748 — a 0.010 difference. The concern did not materialise; temporal
remains the default as the more defensible choice, not because it changes the
numbers.

## Baseline: char n-grams + linear SVM

`python -m classifier.train_baseline`

Burmese has no spaces between words, so whitespace tokenisation fails. Char
3–5 grams (`char_wb`, TF-IDF, 300k features, first 3000 chars) sidestep word
segmentation entirely.

```
baseline most_frequent   acc 0.310   macro-F1 0.043
model                    acc 0.856   macro-F1 0.796    (fit ~76s)
```

| class | F1 | support |
|---|---|---|
| sport | 0.978 | 759 |
| international | 0.958 | 1810 |
| conflict | 0.840 | 1064 |
| environment | 0.837 | 318 |
| health | 0.790 | 100 |
| crime | 0.761 | 437 |
| economy | 0.758 | 424 |
| culture | 0.712 | 81 |
| politics | 0.671 | 776 |
| education | 0.658 | 73 |

### The outlet confound — the result that matters

```
outlet          n     acc  macro-F1
dvb          5060   0.876     0.797
irrawaddy     333   0.814     0.631
bbc           447   0.660     0.491
```

Training is ~80% DVB and the classes **are** DVB's own section names, so the
model can score well by learning DVB's house style rather than topic. BBC's
macro-F1 is ~38% below DVB's, which is what that looks like when it happens.

Report this alongside the headline number. Also note `international` (F1 0.958,
31% of test) is DVB's `နိုင်ငံတကာ` section, where foreign place names give the
class away almost for free — it props up the aggregate accuracy considerably.

### Taxonomy variants tested

A `rights`/`society` class was tried three ways and removed:

| variant | classes | acc | macro-F1 |
|---|---|---|---|
| single `rights` bucket | 11 | 0.846 | 0.748 |
| split `rights`/`society` | 12 | 0.843 | 0.710 |
| **merged into `politics`** | 10 | 0.856 | **0.796** |
| articles dropped | 10 | 0.871 | 0.816 |

Dropping scores highest but wins by deleting 977 hard examples — an easier
problem, not a better model. Merging keeps every article and reflects that in
Myanmar coverage, rights and social reporting is largely political reporting.

Remaining confusions are honest domain ambiguity, not bugs: politics↔conflict
runs both ways because Myanmar's civil war is simultaneously both.

## What this classifier is for

RFA contributes **zero** labelled articles and Myanmar Now 39, because neither
outlet labels topics. So this is a DVB+BBC+Irrawaddy classifier — which makes
its natural use **labelling the 8,806 articles that have no usable topic**,
including all 2,428 RFA articles and nearly all of Myanmar Now.

Train where labels exist, predict where they don't.

---

# Phase 3 — Labelling the whole corpus

Only 29,206 of 38,152 articles carry an outlet label that maps to a topic.
The rest carry *only* format or placement labels — `Video`, `Lead Story`,
`သတင်း` ("news"), `Cartoon` — which say nothing about subject. That gap is not
evenly spread, and that is the problem:

| site | outlet-labelled | predicted | % labelled |
|---|---:|---:|---:|
| dvb | 24,554 | 1,560 | 94% |
| bbc | 2,470 | 287 | 90% |
| irrawaddy | 2,142 | 2,248 | 49% |
| myanmarnow | 40 | 2,411 | 2% |
| **rfa** | **0** | **2,427** | **0%** |
| ALL | 29,206 | 8,933 | 77% |

RFA has *zero* usable labels and Myanmar Now has 40. Any analysis keyed on
topic silently becomes an analysis of DVB plus some Irrawaddy. `predict_unlabelled.py`
fills the gap with model predictions, kept strictly separable from the
outlet labels.

    python -m classifier.predict_unlabelled --data data-vm

## Outputs

    classifier/data/predicted.jsonl   8,933 rows: topic, margin, n_chars, raw_labels
    classifier/data/topics.jsonl     38,139 rows: url -> topic for the whole corpus,
                                     every row tagged source="outlet" | "predicted"
    classifier/data/spotcheck.jsonl      72 rows: stratified sample for human review

38,139 = 38,152 − 5 duplicate URLs − 8 empty bodies. One row per URL, no dupes.

## Confidence: use the margin, not the headline accuracy

LinearSVC gives no probabilities, so confidence is the gap between the best and
second-best class score. Measured on held-out data:

| margin ≥ | accuracy | coverage |
|---:|---:|---:|
| 0.00 | 0.856 | 100% |
| 0.30 | 0.916 | 86% |
| 0.50 | 0.938 | 78% |
| 1.00 | 0.976 | 60% |
| 1.50 | 0.992 | 42% |

**Outlet-labelled ∪ (margin ≥ 0.5) = 34,330 articles, 90% of the corpus**, at roughly 0.94
expected accuracy on the predicted portion. That is the subset to quote in a
write-up; the remaining 10% should be reported as unclassified rather than
silently included.

The 0.856 was measured on DVB/BBC/Irrawaddy — the outlets with labels — so it
does **not** transfer to RFA, which the model never trained on. A 24-article
spot-check across three margin bands (`spotcheck.jsonl`) suggests the margin
*ordering* does transfer: high-margin RFA predictions were correct, and the
errors sat in the low band — e.g. a European heatwave story predicted `economy`
(margin 0.19) and "day student, night soldier" predicted `economy` (margin 0.03)
when both are plainly wrong. Extend that sample before relying on these labels.

## MIN_TRAIN_PER_SITE — the outlet confound, caught in the act

Myanmar Now's 40 labelled articles are **35 economy, 5 politics**. That is not a
sample of Myanmar Now's output; it is the handful of its articles that happened
to carry a mappable label, and the skew is an artefact of *which labels map*.

Trained on them, the model learned "Myanmar Now house style ⇒ economy" and
predicted `economy` for **27.5%** of Myanmar Now's unlabelled articles against a
6.9% corpus prior — a 4× over-fire seen in no other outlet (RFA 7.5%,
Irrawaddy 7.9%, both matching the prior).

Excluding those 40 from training:

| | before | after |
|---|---:|---:|
| myanmarnow `economy` share | 27.5% | **8.5%** |
| myanmarnow `conflict` share | 43.6% | 55.2% |
| myanmarnow median margin | 0.62 | **0.84** |
| held-out accuracy / macro-F1 | 0.856 / 0.796 | 0.856 / 0.796 |

21% of its predictions changed, and the median margin **rose**. Removing
training data normally lowers confidence; it rising is the signature of removing
*contradictory* signal rather than information. Cost on the labelled outlets:
none.

Hence `MIN_TRAIN_PER_SITE = 200`: an outlet with too few labelled articles to
represent itself is excluded from training, while still being predicted. It is
better left out than in as a biased prior on its own style.

## Dataset staleness guard

`taxonomy.py` is read at runtime; `train.jsonl` is a materialised artefact on
disk. Nothing connected them, so editing the taxonomy left a stale dataset that
still loaded, still trained, and still reported plausible numbers for a taxonomy
no longer in force.

This bit. A prediction run got through fit 1 on the 11-class dataset after
rights/society had already been folded into politics; the only tell was a
macro-F1 of 0.748 matching the *superseded* result instead of 0.796.

Fixed structurally rather than by remembering to rebuild:

- `taxonomy.fingerprint()` — sha256 over CLASSES, RESOLUTION_ORDER, MAPPING, DROP
- `build_dataset` stamps it into `classifier/data/manifest.json`
- `dataset_guard.check_dataset()` verifies it and exits 2 with a rebuild command
- wired into `train_baseline` and `predict_unlabelled`

Current fingerprint `124d5b1ddbf12cff` — 10 classes, 147 mapped labels, 42 dropped.

## Remaining caveats

- **RFA accuracy is unmeasured.** Zero outlet labels means no way to score it.
  Hand-labelling ~200 RFA articles would be the single most valuable addition.
- `labelable_but_not_in_dataset = 15` is expected, not a bug: those articles do
  carry mappable labels but were excluded by `build_dataset` for a different
  reason (under 200 chars, or duplicate body).
- Predicted `conflict` runs high for RFA (42.5%) and Myanmar Now (55.2%) against
  a 22.1% prior. Plausible for two outlets built around the post-coup war, but
  unverified — it is also what a `conflict`-biased model would produce.
- Irrawaddy has the lowest confidence of any outlet (median margin 0.35). Not a
  text-length effect: its `Video` articles score *higher* (0.38) than the rest
  (0.26) on 1,286 median chars. It matches its outlet-label macro-F1 of 0.631.

---

# Operational notes

## Where the corpus lives

The crawl ran on an Azure VM (tmux, `~/myhumnews/`). `data-vm/` in this repo is
an rsync backup of it, verified file-for-file per outlet. `data/` holds only the
local Irrawaddy crawl plus 20-article smoke tests — **`data-vm/` is the corpus**,
and every classifier command above points at it.

    rsync -a -e "ssh -i crawler-vm_key.pem" azureuser@<vm>:~/myhumnews/data/ ./data-vm/

Deliberately no `-z`: the bulk is already-gzipped HTML, so compressing again
costs CPU and saves nothing.

## The Irrawaddy must be crawled from a local machine, not the VM

`burma.irrawaddy.com` sits behind Cloudflare, which blocks the Azure IP range
outright — not a headers problem, and not fixable from the VM. Two separate
findings, in order:

1. **curl works where python-requests does not.** Identical headers seconds
   apart: curl gets 200, requests gets 403. Cloudflare scores the TLS/HTTP2
   handshake as well as the headers, so `irrawaddy.py` sets
   `transport = "curl"` and `run.py` selects `CurlFetcher`. This is not
   fingerprint impersonation — curl presents its own, and Cloudflare allows it.
2. **From the Azure VM, curl is blocked too.** The IP itself is refused. The
   Irrawaddy crawl was therefore run on the Mac and rsynced up.

Routing around that block (proxies, address rotation) was considered and not
done: it crosses from "using a client the CDN accepts" into evading a block
someone chose to apply.

So: `python run.py irrawaddy ...` locally, every other outlet on the VM.

## Known data-quality items, all still present in the raw JSONL

| item | count | handled by |
|---|---:|---|
| DVB duplicate JSONL lines (restart race) | 5 | `build_dataset` drops on URL |
| RFA programme-announcement boilerplate | 16 bodies / 117 lines | `build_dataset` drops on body hash |
| empty bodies, corpus-wide | 8 | skipped by both consumers |
| bodies under 200 chars | 55 | `MIN_CHARS` |

None are fixed in place — the raw layer is left exactly as crawled, per the
no-normalisation rule. Both consumers filter them, so they affect only someone
reading `articles.jsonl` directly.

## RFA's 2026 gap was two different things

Jun–Nov 2025 is a genuine publishing collapse. 2026 was **my bug**: RFA changed
URL format from `-MMDDYYYYHHMMSS.html` to `/YYYY/MM/DD/slug/` and the `.html`
filter silently dropped every 2026 article. Worse, discovery still reported
success — those days were counted as `empty_days`, indistinguishable from the
real gap. Both regexes now in `rfa.py`.

The lesson, recorded because it cost a full re-crawl: a URL filter that matches
the samples you happened to test is not the same as one that matches the site.

---

# Phase 4 — Hand-labelling RFA to measure it

RFA's 2,427 topics were predictions with no ground truth behind them, and the
0.856 headline was measured on DVB/BBC/Irrawaddy. This measures RFA directly.

    python -m classifier.sample_for_labelling --n 200   # draw blind sample
    python -m classifier.label_web                      # label in a browser
    python -m classifier.score_labels                   # measure

## Simple random, not stratified — and why that costs something

Only a random sample gives an unbiased accuracy estimate. Stratifying by the
model's predicted topic would measure per-class precision but **cannot measure
recall at all**: you would never sample the articles the model wrongly moved
*out* of a class, because you are sampling by where it put them.

The cost is a thin tail. At RFA's predicted distribution, 200 random articles
yield roughly conflict 86, politics 51, crime 17, environment 15, economy 14,
international 10, and three or fewer each of culture/health/education/sport.
That measures the top six — 96% of RFA — properly and leaves the tail
unmeasured. `score_labels` marks any class with support under 10 as noise
rather than printing it as a result. For tail numbers, draw a *separate* top-up
sample; do not dilute this one.

## The sample is blind, deliberately

`label_task.jsonl` contains no predictions. Showing the model's guess while
labelling anchors the labeller toward agreeing, and an accuracy measured against
a visible prediction is not a measurement. `score_labels.py` joins the two by
url afterwards.

The draw is seeded (2026), so it is reproducible and a re-run will not silently
produce a different 200.

## The question this actually answers

The high-confidence subset — outlet-labelled ∪ (margin ≥ 0.5), 90% of the corpus — rests on
a margin threshold calibrated on the labelled outlets. A 24-article spot-check
hinted the ordering transfers to RFA, but 24 is a hint.

`score_labels` prints accuracy in three **disjoint** margin bands. If RFA
accuracy does not rise across them, the 90% subset is not trustworthy and that
claim must be withdrawn. If it does, the subset is evidenced rather than
assumed. Either way it stops being a guess.

At n=200 the 95% Wilson interval is about ±5 points, so the report always prints
the interval alongside the point estimate.

## Label in the browser, not the terminal

**Terminals cannot render Burmese.** Burmese stacks combining marks above, below
and around its consonants — U+1037 dot-below, U+103A asat, medials — and a
terminal that gives each codepoint its own fixed cell draws that as disconnected
fragments. Labelling text you cannot read accurately produces noise, not ground
truth, which would defeat the whole exercise.

So `label_web.py` serves a local page and the browser hands the text to the OS
text engine, which shapes it correctly. This Mac has Noto Sans Myanmar in
`/System/Library/Fonts`, so no webfont and no network are needed.

    python -m classifier.label_web      # opens http://127.0.0.1:8765

The page sets `line-height: 2.25` on the body text (at 1.5 the stacked marks
collide with the row below) and `overflow-wrap: anywhere` (Burmese has no
inter-word spaces, so the browser finds no break opportunity and the paragraph
would otherwise overflow horizontally).

A **local** server rather than a published page for two reasons: the labels must
land in the same `classifier/data/labels.jsonl` that `score_labels` reads, and
there is no reason to send 200 news articles to a third party in order to read
them in a nicer font. It binds `127.0.0.1` only.

`label_cli.py` is the terminal equivalent and remains in the repo — it is useful
over SSH, where a browser is not available. The two write the identical format
and share the same digit keys, so you can start in one and finish in the other.

## Notes on both tools

- **Resumable.** Each label is appended, flushed and fsynced immediately, so
  Ctrl-C or a closed laptop loses nothing. Same rule the crawler follows.
- Keys `1`–`9`,`0` are the ten classes in `taxonomy.CLASSES` order; `a` flags
  ambiguous, `s` skips, `b` undoes, `?` opens the rules.
- **`classifier/LABELLING_GUIDE.md`** holds the decision rules, every one of
  them derived from what the outlet labels actually do rather than from intuition.
  The browser tool shows a condensed version under `?`.
- Skips (programme announcements, boilerplate) are excluded from scoring and
  reported separately — they are a finding about the corpus, not model errors.
- ~200 articles at 15–20s each is roughly an hour, in as many sittings as you
  like.

## `a` — ambiguous is not the same as skip

Two different non-labels, kept apart because they are findings about different
things:

- **`s` skip** — not an article (programme announcement, boilerplate). A finding
  about the *corpus*.
- **`a` ambiguous** — a real article the taxonomy genuinely does not decide. A
  finding about the *taxonomy*.

Guessing on an ambiguous article is indistinguishable from a model error and
quietly corrupts the measurement, so `score_labels` reports a **bound** instead
of dropping them: best case the model got them all right, worst case all wrong,
true accuracy in between. Above 10% ambiguous it says so explicitly, because at
that point the taxonomy needs work before the headline figure means much.

### The case that motivated it

Industrial mining accidents have **no consistent outlet label**. The same event —
a Hpakant mud-pond collapse with casualties — appears as three different classes
in the training data:

    [crime      ] ဖားကန့်တွင် နွံကန်ပြိုကျမှုကြောင့် သေဆုံးသူ ၈ ဦးအထိရှိလာ
    [politics   ] ဖားကန့်မှာ နွံကန်ပြိုကျပြီး လူ ၁၀ ဦးထက်မနည်း သေဆုံး
    [environment] ဖားကန့်တွင် နွံကန် ပေါက်ပြီး ကားလမ်းပိတ်၊ သွားလာရေးခက်ခဲ

The root cause is a genuine gap: `environment` is defined as "weather, climate,
earthquakes, disasters", and an industrial accident is not a natural disaster,
so the taxonomy has nowhere to put it. Decide deliberately after the
measurement — add a class, or write an explicit rule — rather than papering over
it with guesses now.

(Military *attacking* a mining area is unambiguous: the military is the actor,
so `conflict`. That is 45% of Hpakant titles.)

### A known taxonomy inconsistency

`taxonomy.py` maps စစ်ဘေးရှောင် (war-displaced) → **conflict** but
ဒုက္ခသည်များ (refugees) → **politics**. Both mean displaced people, so otherwise
identical articles receive different outlet labels depending on which word the
outlet used. Found while writing the labelling guide; not yet fixed, because
changing it invalidates the dataset and the fix should be chosen deliberately.

---

# Phase 4 results — RFA measured

200 articles hand-labelled, two passes. **This is the first real measurement of
the model outside the outlets that carry labels, and it is not good news.**

## The headline

    scored 177   skipped 17 (non-articles)   ambiguous 6

    accuracy  0.633   95% CI [0.560, 0.700]
    ambiguous bound   [0.612, 0.645]   -- 3%, narrow enough that the figure stands

    for comparison:  dvb 0.875   irrawaddy 0.814   bbc 0.663

RFA is the **worst outlet in the corpus**, below BBC. The corpus-wide 0.856 does
not describe it and must never be quoted as if it does.

## Two labelling-convention issues inflate the error count

| treatment | accuracy | n |
|---|---:|---:|
| as labelled | 0.633 [0.560, 0.700] | 177 |
| excluding `international` | 0.700 [0.625, 0.766] | 160 |
| + conflict wins politics ties | 0.844 [0.780, 0.892] | 160 |

**`international` (17 articles, model got 0 right).** 15 of the 17 are Myanmar
stories with a foreign actor — a Vietnamese PM commenting on Myanmar,
Switzerland sanctioning MOGE, an ASEAN envoy meeting ASSK. The outlet label
`international` means foreign news with *no* Myanmar angle: only 1.6% of 8,674
such titles mention Myanmar. So the model's `politics` calls were
right and these are not model errors. Correcting them is defensible because the
convention is documented and measurable.

**politics → conflict (23 of 65).** `RESOLUTION_ORDER` puts conflict above
politics, so where both genuinely apply the taxonomy favours conflict — and the
model. Several are clear-cut ("NUG minister says battles will begin in Yangon
and Bago"); others are not ("JFM exposes a Singapore firm supplying fuel to the
junta" is arguably economy). Resolving them needs per-article review, so **0.844
is an upper bound, not a finding.**

**The defensible figure is ~0.70, with a plausible range of 0.63–0.84.** The
conclusion is the same at either end: RFA is substantially worse than DVB.

## Per class — where it works and where it does not

    environment   P 0.875  R 0.933  F1 0.903   n=15   solid
    conflict      P 0.600  R 0.875  F1 0.712   n=48   over-fires, catches most
    economy       P 0.579  R 0.846  F1 0.688   n=13
    politics      P 0.667  R 0.492  F1 0.566   n=65   misses half
    crime         P 0.583  R 0.538  F1 0.560   n=13
    international P 0.000  R 0.000  F1 0.000   n=17   total failure

`international` fails completely on RFA, and not only through the convention
issue: the model predicted it **once** in 177 articles, and wrongly. It
predicts `international` for just 5.2% of RFA against a true rate near 9.6%.
The detector was trained on DVB, where 6,406 of 6,865 `international` articles
came from one outlet's foreign desk — it learned that desk's house style, not
the concept.

## The margin threshold: ordering transfers, calibration does not

    disjoint bands:   low <0.3   0.438  (n=32)
                      mid        0.586  (n=70)
                      high >=1.0 0.760  (n=75)

Monotonic, so **filtering by margin genuinely helps** — a high-margin RFA
prediction is worth roughly 1.7x a low-margin one.

But the *level* does not transfer:

| margin ≥ 0.5 | labelled outlets | RFA |
|---|---:|---:|
| accuracy | 0.938 | **0.685** |

### Withdrawn

Phase 3 claimed: *"outlet-labelled ∪ (margin ≥ 0.5) = 34,330 articles, 90% of the corpus,
at roughly 0.94 expected accuracy."* **That is wrong for RFA**, where the same
threshold yields 0.685. The 0.94 was calibrated on DVB/BBC/Irrawaddy and silently
assumed to carry over. It does not. A 24-article spot-check had hinted the
ordering transferred, and I over-read that as the calibration transferring too.

Corrected statement: use margin to **rank** confidence within an outlet, and
quote accuracy **per outlet**, never one figure across all five.

## What to report in the write-up

- Topic figures for **DVB and Irrawaddy** are on solid ground (0.875, 0.814).
- **BBC (0.663) and RFA (~0.70)** should be reported with their own numbers and
  the caveat attached, or excluded from topic-level claims.
- **Myanmar Now is unmeasured.** It has 40 outlet labels, excluded from training
  by `MIN_TRAIN_PER_SITE`, and no hand-labelled sample. Treat its 2,411
  predicted topics as unvalidated.
- Do not quote a single corpus-wide topic accuracy. The spread is 0.663 to
  0.875 across measured outlets.

## Fix with the best return, if this gets taken further

Hand-label ~200 **DVB** articles. It sounds redundant, but DVB's outlet labels are
its own section names, so the 0.875 is partly circular — it measures agreement
with DVB's desk conventions, not topical accuracy. An independent DVB
measurement would tell you how much of the headline number is real. Second
priority: ~100 Myanmar Now articles, the only wholly unmeasured outlet.

---

# How the hand labels are used

Three possible uses, and they are not interchangeable.

## 1. Evaluation — what they were drawn for

`score_labels.py` joins them to `predicted.jsonl` by url and reports accuracy,
per-class figures and the margin bands. This is the only unbiased measurement of
RFA that exists.

## 2. Correcting the corpus — `apply_hand_labels.py`

    python -m classifier.apply_hand_labels

A hand label is the strongest evidence in the project: a human read the article
and decided. Before this step `topics.jsonl` still carried the model's guess for
all 200, including **65 that had already been shown wrong** — a known-wrong value
sitting in the deliverable.

Applied:

    177  topics replaced with the human's label   source = "hand"
     65  of those were corrections
     17  non-articles -> topic = null             source = "hand"
                                                  reason = "not_an_article"
      6  ambiguous -> prediction kept, flagged    reason = "ambiguous"

`topics.jsonl` is now 29,206 outlet + 8,739 predicted + 194 hand, with 17 rows
carrying no topic because they are not articles. The three sources stay
distinguishable, so any analysis can filter to verified rows or exclude flagged
ones.

Biggest corrections, model → human:

     23  conflict -> politics
     12  politics -> international
      3  conflict -> crime
      3  culture  -> politics
      3  economy  -> politics

## 3. Training — deliberately NOT done

Adding them to training would be a mistake, and not a small one. These 200
articles are the only unbiased measurement of RFA in existence. Train on them and
they stop being a test set: the model would score well because it had seen them,
and there would be no way left to tell whether it had genuinely improved.

**The choice is exclusive — you cannot both train on and measure with the same
200 articles.** 177 labels against 29,166 training examples would also barely
move the model; the measurement is worth far more than the marginal training
signal.

If hand labels for training are wanted, draw a **separate** sample with a
different seed. Never reuse the evaluation set.

    python -m classifier.sample_for_labelling --n 300 --seed 77 \
        --out classifier/data/train_task.jsonl

---

# Phase 5 — DVB measured independently, and what it overturned

200 held-out DVB articles hand-labelled (seed 31, blind to both the outlet label
and the prediction). DVB has outlet labels, so all three comparisons were
available — and the third is the one that mattered.

## The circularity test

    model vs hand   0.604  [0.534, 0.670]   <- the model's real accuracy
    outlet vs hand   0.645  [0.576, 0.708]   <- how good DVB's OWN labels are
    model vs outlet   0.853  [0.797, 0.896]   <- the number usually quoted

**DVB's own section names agree with a human reading only 64.5% of the time.**
So the 0.875 headline was never topical accuracy; it was agreement with a noisy
label source.

The model sits at **94% of its ceiling** (0.604 / 0.645). It cannot do much
better while trained on these labels. **The bottleneck is the label source, not
the model** — more features, a segmenter or a transformer would all be chasing
the wrong 6%.

                    outlet right   outlet wrong
    model right            114            5
    model wrong             13           65

Both wrong on 65 of 197 (33%). That cell is a verdict on the taxonomy.

## Correction: RFA is NOT the worst outlet

I reported after Phase 4 that "RFA is the worst outlet in the corpus, below
BBC". **That was wrong**, and wrong in an avoidable way — I compared RFA's
*independent* measurement against DVB's *circular* one.

    model vs hand, rfa   0.633  [0.560, 0.700]   n=177
    model vs hand, dvb   0.604  [0.534, 0.670]   n=197

Measured the same way, the intervals overlap heavily and the two are
**indistinguishable**. The apparent DVB superiority was entirely an artefact of
grading the model against the labels it trained on. Phase 4's conclusion stands
only in its narrow form: RFA accuracy is ~0.63. The comparison to DVB does not.

## Root cause: `international` is a placement label, not a topic

46% of the outlet/hand disagreements have `outlet label = international`, and **0 of those
32 mention Myanmar.** They are foreign stories labelled by subject by a human
and filed under the foreign desk by DVB:

    you=environment   Venezuela earthquake, ~1,000 dead
    you=conflict      overnight Russian strikes kill 13 in Ukraine
    you=environment   Poland's Vistula falls as temperatures climb
    you=health        bird flu confirmed near Esperance, Australia
    you=economy       US-Iran peace pushes crude under $80

`နိုင်ငံတကာ` is DVB's **foreign desk** — where a story was placed, not what it
is about. `RESOLUTION_ORDER` puts `international` **last** ("only if nothing
else applies"), so by the taxonomy's own rule labelling by subject is correct
and the section name is what is wrong.

This is a design error of mine. `_DROP_RAW` already excludes placement labels —
`Lead Story`, `Video`, `Cartoon` — on exactly this reasoning. A foreign desk is
placement too, and I mapped it as a topic anyway.

It is also the **largest class, 8,674 articles = 29.7% of the dataset**, and it
is **not automatically recoverable**: all 8,674 carry `international` as their
*only* mappable label, so there is no second category to fall back on.

## The honest numbers, once `international` is set aside

    excluding rows where the outlet label or hand is international   (n=119)
      model vs hand   0.731  [0.645, 0.803]
      outlet vs hand   0.798  [0.717, 0.861]

**On genuine topics the corpus labels are sound (0.798) and the model gets
0.731** — at 92% of ceiling. That is the defensible result. The 0.604 is
depressed by a 30% class that is not a topic, and the 0.875 was inflated by the
same class being easy to detect.

Per class, `sport` is the one that is unambiguously clean:

    sport          P 1.000  R 0.964  F1 0.982   n=28
    conflict       P 0.867  R 0.650  F1 0.743   n=40
    crime          P 0.750  R 0.480  F1 0.585   n=25
    international  P 0.508  R 0.696  F1 0.587   n=46
    environment    P 0.727  R 0.276  F1 0.400   n=29
    politics       P 0.192  R 0.385  F1 0.256   n=13

`environment` recall 0.276 is the same bug from the other side: the model sends
11 of 29 environment articles to `international`, because that is what DVB's
labels taught it to do with foreign weather and disasters.

## What to do about `international` — a decision, not a fix

Three options, none free:

1. **Drop the class.** Taxonomy-consistent, but all 8,674 articles leave the
   dataset (29,206 -> 20,532) since they carry no other mappable label. The
   model also loses any ability to describe foreign coverage.
2. **Rename it to what it is** — `foreign`, a placement/origin flag — report it
   separately, and exclude it from macro-F1 as if it were a topic. Cheapest and
   honest. Topic accuracy becomes the 9-class figure.
3. **Make it a second axis**: every article gets `(topic, is_foreign)`. The
   principled fix, but it needs the 8,674 relabelled by subject, which no
   automatic rule can do.

**Recommended: 2 for this write-up, 3 if the project continues.** Report "topic
accuracy 0.731 on 9 classes" plus "foreign-desk detection" separately, rather
than one blended number that is neither.

## What to report now

- **Do not quote 0.875, or 0.856 corpus-wide.** Both measure agreement with
  DVB's section names.
- **Topic accuracy, DVB: 0.731** [0.645, 0.803], 9 classes, independently
  measured.
- **RFA: 0.633** [0.560, 0.700]. Statistically indistinguishable from DVB.
- **Label quality is itself a finding**: DVB's section names match a human 64.5%
  overall, 79.8% once the foreign desk is excluded.
- BBC, Irrawaddy and Myanmar Now remain measured only circularly or not at all.

---

# Phase 6 — the HumSet gold set (383 articles, two passes)

`classifier/data/humset_gold.jsonl` — **383 articles, 7 HumSet labels, multi-label.**
This is the hand-checked test set the proposal's macro-F1 metric is measured on.
Built from two passes, both preserved:

| pass | annotator | n | file |
|---|---|---|---|
| 1 | HsuLinHtet | 383 (all) | `annot/labels-HsuLinHtet.jsonl`, backup `annot.pass1-20261015` |
| 2 | ZweHtetAung | 85 (the `other` subset) | `annot-other/labels-ZweHtetAung.jsonl` |

Every row carries `pass`, `annotator`, and — for re-labelled rows — `pass1_labels`,
so the merge is reversible and auditable.

## Why there was a second pass

Pass 1 put **85 of 383 (22.2%)** articles in `other`. Reading all 85 showed the
label was over-applied: Myanmar political and economic coverage was landing in
`other` when it belongs in `context`.

**Root cause was the guide, not the annotator.** `HUMSET_GUIDE.md` defined
`other` as "no humanitarian relevance" and left *relevance* undefined, so
"no acute emergency" was a reasonable reading of it. The guide now leads with
the Myanmar-relevance test, a table of worked examples both ways, and the
reason it matters:

> A false `other` silently removes an article from the whole system. When
> genuinely torn between `other` and `context`, choose `context`.

Pass 2 served **all 85**, not just the 24 I had flagged. Serving only my picks
would have made it a confirmation of my judgement rather than an independent
one — and my judgement was the thing in question.

## Result of pass 2

```
still other   69  (81.2%)        rfa  n= 20  changed 13  (65.0%)
changed       16  (18.8%)        dvb  n= 65  changed  3  ( 4.6%)
```

**The problem was almost entirely RFA.** DVB's `other` articles were genuinely
`other` — sport, entertainment, foreign news with no Myanmar angle — and 94%
survived. RFA's did not: two thirds were Myanmar politics, economy, sanctions
and labour stories that belong in `context`. Pass 1 was not applying `other`
carelessly; it was applying a defensible reading of a vague guide, and that
reading only did damage where the outlet covers Myanmar policy closely.

## My 24 proposals scored against pass 2

I was right about the direction and unreliable about the specifics:

| | n | pass 2 also moved it | exact label-set match |
|---|---|---|---|
| high confidence | 16 | 10 (63%) | 6 (38%) |
| medium | 4 | 3 | 2 |
| low | 4 | 2 | 1 |
| **all** | **24** | **15 (62%)** | **9 (38%)** |

And of the **61 I judged correctly `other`, 60 were kept (98.4%)** — one DVB
story about Yangon Chinatown food stalls reopening went to `context`.

So: good at ruling `other` **in**, mediocre at ruling it **out**, and poor at
picking the replacement set. **My stated confidence did not predict correctness**
— 63% on "high" vs 50% on "low" is not a usable signal at n=24. Six of the nine
I got exactly right were plain single-`context`; every case needing a second
label (`humanitarian_access`, `shock_event`) I got wrong in one direction or the
other. This is the argument against letting an LLM pre-label without review
landing in the gold set, and it is worth reporting as such.

## Final distribution

```
label                        pass1  final  delta    % of 383
context                        173    186    +13      48.6%
shock_event                    177    179     +2      46.7%
casualties                      85     85      0      22.2%
other                           85     69    -16      18.0%
displacement                    46     46      0      12.0%
humanitarian_access             12     16     +4       4.2%   THIN
information_communication        6      6      0       1.6%   THIN
```

mean 1.53 labels/article · cardinality 1→212 2→141 3→27 4→3

Integrity holds: 0 empty sets, 0 `other`-with-other-labels, and
**0 `casualties` without `shock_event`** across all 383 — the one structural
rule the guide imposes, respected by both annotators independently.

## Two labels are not measurable and must not be reported as results

`humanitarian_access` (16) and `information_communication` (6) are **below 20
examples**. A per-label F1 on 6 positives moves by 0.17 for a single article.
Report them as **thin support with the count stated**, never as a score. The
scorer prints `thin` next to them and reports macro-F1 twice for this reason.

# Phase 6 — `score_multilabel.py`

```
python -m classifier.score_multilabel --gold classifier/data/humset_gold.jsonl
python -m classifier.score_multilabel --gold <gold> --pred <model_dump>
python -m classifier.score_multilabel --gold <annotA> --pred <annotB> --agreement
```

Rows match on **`url`, never on index**. The three label stores were written
with different index bases — `other_review.jsonl` uses the original 1–383
index while the re-labelling task is re-indexed 1–85 — and matching on `i`
mis-aligned them once already during this merge.

What it reports and why `accuracy` is absent: every intuitive definition of
accuracy for multi-label output flatters the model.

- **per-label P/R/F1** — seven independent binary problems; the proposal's unit.
- **macro-F1, reported twice** — over all 7, and over the 5 with support ≥ 20.
  The gap between them is the finding, not a nuisance: the all-7 number
  inherits the variance of a 6-example label.
- **micro-F1** — pools decisions, always higher here. Next to macro, never instead.
- **exact match + Wilson CI** — the harsh, honest number a user experiences.
- **Hamming loss** — diagnostic only. Predicting *nothing at all* scores ~0.22
  on this set, so a small value means little.
- **a floor**: always predict the two commonest labels → **macro-F1 0.184,
  exact match 0.097**. Any model must clear this to have done anything.

`--agreement` is symmetric (no ground truth): Jaccard, per-label Cohen's kappa,
and the swap table. It guards one trap it hit immediately — on the 85-article
subset **every kappa is 0.000, and that is correct, not disagreement**. The
subset was *selected* on pass 1 saying `other`, so pass 1 is constant, chance
agreement already equals observed agreement, and kappa is 0 by construction.
The scorer now detects a constant rater and says so in the output. **A real
inter-annotator figure needs a double-labelled UNSELECTED sample** — which this
project does not yet have.

# Phase 7 — TF-IDF multi-label baseline (`train_humset.py`)

```
python -m classifier.train_humset                      # 5-fold CV on the gold set
python -m classifier.train_humset --train <llm.jsonl>  # train on pre-labels, test on gold
```

The TF-IDF arm of the proposal's TF-IDF-vs-XLM-R ablation. One-vs-rest
`LinearSVC` over `char_wb` 3–5 gram TF-IDF, `class_weight="balanced"`,
title + body.

## Headline

```
macro-F1 (support >= 20, n=5)   0.755        <-- the number to report
macro-F1 (all 7 labels)         0.539
micro-F1                        0.773
exact match                     0.569   95% CI [0.519, 0.618]
Jaccard                         0.718
Hamming loss                    0.0966

floor (always predict context+shock_event)  0.258  -->  beaten by +0.497
```

| label | P | R | F1 | support |
|---|---|---|---|---|
| shock_event | 0.879 | 0.855 | **0.867** | 179 |
| context | 0.754 | 0.823 | **0.787** | 186 |
| other | 0.828 | 0.696 | **0.756** | 69 |
| casualties | 0.762 | 0.718 | **0.739** | 85 |
| displacement | 0.703 | 0.565 | **0.627** | 46 |
| humanitarian_access | 0.000 | 0.000 | **0.000** | 16 — thin |
| information_communication | 0.000 | 0.000 | **0.000** | 6 — thin |

## This is a supervision-starved number, not the proposal's TF-IDF arm

The proposal trains on **LLM pre-labels at corpus scale (38,152 articles)** and
tests on these 383. Those pre-labels do not exist yet — there is no API key in
the environment, so corpus-scale pre-labelling is **blocked on a credential**,
not on code. `--train` is already wired for the moment one arrives.

So 0.755 is what char n-grams achieve from **383 examples**, which is a floor
with real features rather than the intended arm. Report it as such.

## Why cross-validation, not a train/test split

383 articles. An 80/20 split puts `information_communication` (6 positives) at
an expected 1.2 in the test fold; the resulting number would be noise. 5-fold
CV tests every article exactly once, so the scores are over all 383. It does
**not** fix the underlying scarcity — a 6-positive label has ~4.8 per training
fold.

Folds are grouped by label-set signature (sklearn has no multi-label
stratifier); signatures appearing fewer than 5 times collapse into `RARE`.

## The variance work, and why C=0.5 is not cherry-picked

A 30-point sweep over analyzer, n-gram range, `max_features` and `C` spanned
**0.686 – 0.758**. Picking the maximum off that sweep would be selection bias
at n=383, so the question is which differences survive a paired test.

```
within one CV run, per-fold macro-F1*   0.719  0.817  0.671  0.734  0.767
                                        sd 0.055   range 0.146
across 5 different CV splits   C=0.5  mean 0.747  sd 0.010
                               C=1.0  mean 0.732  sd 0.013
paired difference per split     +0.011 +0.014 +0.012 +0.021 +0.018
                                mean +0.015  sd 0.004  —  5/5 same sign
```

**Fold-to-fold noise (sd 0.055) is five times the C effect (0.015).** A single
sweep could not have justified changing `C`; the paired-across-splits test
could, and C=0.5 won 5/5. That is now the default, with the reasoning at the
definition site.

Two findings fell out of the same sweep:

- **`max_features=300_000` is a no-op here.** `char_wb` 3–5 with `min_df=2`
  over 383 articles builds only **65,955** n-grams. Kept at 300k for the
  corpus-scale run, where it will bind.
- **Naive word features are clearly worse: 0.694 vs 0.758.** Burmese is written
  without spaces between words, so whitespace tokenisation produces garbage
  features. This supports treating "segment words" as a weak ablation for the
  TF-IDF arm — but note what it does and does not show. It shows *whitespace
  tokenisation* loses. It does **not** show a real segmenter would lose; that
  remains untested.

## One-vs-rest produces invalid output, and the repair barely matters

Seven independent binary decisions do not know that `other` is exclusive, or
that every article has at least one label. Raw output gave **32 empty sets
(8.4%)** and **3 `other`+something (0.8%)**. The repair fills an empty set with
the highest-scoring label and resolves an `other` conflict by score.

It is worth **+0.000 macro-F1** — it moves exact match (0.530 → 0.569) and
Jaccard, because it fixes whole-set correctness, but it does not make the
per-label decisions better. Keep it for output validity, do not claim it as a
modelling gain.

## RFA is harder than DVB, and this time the comparison is fair

```
dvb   n=200   macro-F1 0.772   exact 0.635
rfa   n=183   macro-F1 0.633   exact 0.497
```

Unlike the Phase 5 general-news case — where an apparent RFA/DVB gap turned out
to be an independent measurement compared against a circular one — both sides
here are scored against the same hand-labelled gold set by the same procedure.
The gap is real. RFA's `other`-vs-`context` boundary was also where pass 1's
labelling broke down (65% of its `other` articles were re-labelled, against
4.6% of DVB's), so RFA's humanitarian/political boundary is genuinely the
harder one, for annotator and model alike.

## The dominant error is a boundary, not a scatter

```
19 (27.5% of other)                other -> context
 9 ( 4.8% of context)              context -> other
 8 ( 4.5% of shock_event)          shock_event -> context
 7 ( 3.8% of context)              context -> shock_event
 6 (37.5% of humanitarian_access)  humanitarian_access -> context
```

`context` absorbs errors from every direction — it is the largest label and the
vaguest. The `other`↔`context` pair alone is 28 of the errors. That is the
**same boundary the annotation guide got wrong** in pass 1, which is some
evidence the distinction is intrinsically hard rather than badly defined.

## The two thin labels are not hard — they are below the floor

The obvious reading of `humanitarian_access` F1 0.000 and
`information_communication` F1 0.000 is "these categories are hard". A learning
curve says otherwise. Each well-supported label was retrained with its
positives capped at k in the **training** folds only (test folds untouched),
mean of 3 subsamples:

| label | full | k=5 | k=10 | k=15 | k=20 | k=30 | k=45 |
|---|---|---|---|---|---|---|---|
| shock_event | 0.860 | 0.026 | 0.223 | 0.433 | 0.556 | 0.678 | 0.772 |
| context | 0.778 | 0.000 | 0.048 | 0.101 | 0.210 | 0.387 | 0.627 |
| casualties | 0.744 | 0.008 | 0.103 | 0.241 | 0.379 | 0.539 | 0.675 |
| displacement | 0.627 | 0.000 | 0.055 | 0.148 | 0.343 | 0.505 | 0.627 |

**Every label collapses to ~0.00 at k=5 and stays under 0.25 at k=15** —
including `shock_event`, the easiest label in the set at full support. At
`humanitarian_access`'s actual 16 positives the curve predicts 0.10–0.45; at
`information_communication`'s 6, it predicts ~0.00.

So the two zeros are a **data quantity** result, not a difficulty result, and
they come with a concrete target:

```
F1 ~ 0.35-0.55   needs ~20-30 positives
F1 ~ 0.5 -0.77   needs ~45 positives
```

`humanitarian_access` needs roughly **15–30 more** positives,
`information_communication` roughly **25–40 more**. At their 4.2% and 1.6% base
rates, finding those by uniform random labelling would mean labelling ~700 and
~1,900 more articles. That is the argument for targeted sampling rather than
more of the same.

# Phase 7 — `prelabel_llm.py` (built, blocked on a credential)

```
python -m classifier.prelabel_llm --estimate                    # no API calls
python -m classifier.prelabel_llm --validate-set                # measure the labeller first
python -m classifier.prelabel_llm --uniform 8000                # training pool
python -m classifier.prelabel_llm --target humanitarian_access --n 600
```

The proposal's "Label (LLM pre-labels)" step. Output is **training data only**
and must never be merged into `humset_gold.jsonl` — the headline metric is
macro-F1 on a hand-checked test set, and gold contaminated with model labels
measures the labeller instead of the classifier.

Stdlib `urllib`, no new dependency. Resumable (skips URLs already in the output
file), threaded, prompt-cached system message, and it re-applies the guide's two
hard rules server-side: `other` exclusivity, and `casualties` implies
`shock_event`.

**Status: there is no `ANTHROPIC_API_KEY` in the environment, so this is blocked
on a credential, not on code.** `--estimate` runs without one.

## Cost, measured from the corpus

38,093 articles have usable bodies (≥200 chars), mean 2,321 chars.

**Burmese tokenizes badly** — roughly 1.45 tokens per character against ~0.25
for English — so an average article is ~3,400 input tokens, not the ~580 an
English article of that length would cost. That single fact drives the budget:

| scope | input tokens | est. cost |
|---|---|---|
| whole corpus (38,093) | 133.4M | **$424** |
| uniform 8,000 | 28.2M | **$90** |
| uniform 2,000 | 7.1M | **$22** |
| the 383 gold articles (to measure the labeller) | — | **~$4** |

at $3/$15 per Mtok. Prices are hardcoded in the file and will go stale.

## Do `--validate-set` first

`--validate-set` labels the **383 gold articles** and scores the LLM against the
hand labels for ~$4. That number is the **ceiling** for anything trained on
these pre-labels, and it is worth knowing before spending $90–$424.

There is specific reason for caution. On the `other`-subset review in Phase 6,
my own LLM judgement on these exact labels got the **direction** right (62% of
flagged articles were genuinely mislabelled) but the **exact label set** right
only **38%** of the time, and stated confidence did not predict correctness.
Pre-labels at that quality cap a classifier well below the 0.755 the gold set
already supports.

## Targeted sampling, and the assumption it rests on

`--target LABEL` trains the one-vs-rest scorer on all 383 gold articles for that
label, scores the corpus, and sends the top `--n` by decision margin. Chosen
over keyword seeds so the method does not bake in guesses about Burmese
vocabulary.

**The caveat is load-bearing:** for the two thin labels the gold-trained
classifier scores F1 0.000, so its ranking is *unverified*. Phase 4 found that
margin **ordering** transfers across outlets where **calibration** does not, so
ranking may well beat the 4.2%/1.6% base rate — but that is an assumption. The
script prints the realised hit rate against the base rate after every `--target`
run, so the assumption gets tested on the first 600 articles. If they come back
near the base rate, the ranking added nothing and keyword seeding is the
fallback.

# Phase 8 — free weak supervision from section labels: a negative result

```
python -m classifier.weak_supervision --audit      # rule precision vs gold
python -m classifier.weak_supervision --evaluate   # gold-only vs gold+weak
```

With LLM pre-labelling priced out (~$424, no budget), the obvious free
substitute is the outlets' own section labels: they cover all 38,152 articles
and cost nothing. Rule precision measured against the 383 gold articles looked
genuinely promising:

| rule | precision | corpus pool |
|---|---|---|
| sport\|football → `{other}` | **100%** (27/27) | 2,966 |
| politics → `context` | 92% (22/24) | 5,147 |
| military/civil war → `shock_event` | 91% (30/33) | 5,292 |
| crime → `shock_event` | 87% | 1,937 |
| weather/environment → `shock_event` | 86% | 1,183 |
| economy → `context` | 75% | 1,367 |

The `other` rule is special: because `other` is **exclusive**, a sport article's
full label set is known to be exactly `{other}`, so those 2,966 articles carry
*complete* supervision — positive for `other`, negative for the other six.

Supervision is tracked **per label** (`pos` / `neg` per row), because a weak
rule asserts one label and says nothing about the rest; training one-vs-rest on
such rows as if the unasserted labels were negative would inject false
negatives at scale.

## It does not work, and the dose-response curve says why

```
weak pool size      macro-F1*   exact match
none (gold only)       0.755       0.569
400                    0.759       0.546
1,200                  0.745       0.509
3,000                  0.729       0.449
16,087 (all)           0.702       0.290
```

Monotonic degradation in proportion to volume. Per label at full pool:

```
context        0.787 -> 0.731   -0.056      shock_event  0.867 -> 0.717  -0.150
other          0.756 -> 0.641   -0.115      displacement 0.627 -> 0.674  +0.047
casualties     0.739 -> 0.749   +0.009
```

**The mechanism is dilution, not label noise.** If 8–14% wrong labels were the
problem, 400 weak rows would hurt proportionally; instead small amounts are
neutral and harm scales with volume. `shock_event` received 11,217 weak rows
against ~306 gold rows per fold — **36:1**. At that ratio the model stops
learning "is this a humanitarian shock event" and learns "is this in the
military section." It fits the proxy.

Two confirmations of the diagnosis:

- **Every label that gained weak *positives* got worse. Both labels that gained
  only weak *negatives* improved** (`displacement` +0.047, `casualties` +0.009),
  and those negatives came from the 100%-precise sport rule.
- **`other` fell 0.115 despite both its rules being ~100% precise.** 2,966 sport
  positives against ~55 gold ones narrows the learned concept to *sport*, when
  gold `other` also contains entertainment and foreign news.

## Why this was predictable, in hindsight

This is the **same disease as Phase 5**: section labels are **placement, not
topic**. `international` was a foreign desk rather than a subject, and
`မြန်မာသတင်း` ("Myanmar news", 13,980 articles) is a desk too. The project hit
this once from the taxonomy side and once from the weak-supervision side.

**Verdict: drop it, do not tune it.** At its best (cap 400, roughly 1:1 with
gold) it is a wash — +0.004 macro-F1 against a 0.023 fall in exact match. The
corpus metadata does not contain humanitarian structure, and there is no free
shortcut around hand labels for this task. The dose-response curve is the
reportable finding.

# Phase 8 — what hand labelling still owes

Round 2 (`sample_thin_labels.py`, 360 targeted articles) fixes **training** data
for the two thin labels. It does **not** fix measurement:

```
label                       gold test pos   rate    random articles for 30 test pos
humanitarian_access                   16   4.2%    718   (335 more)
information_communication              6   1.6%  1,915 (1,532 more)
```

Round 2 is an enriched, non-representative sample and belongs in a training
pool. The gold set still holds 16 and 6 positives, so no amount of training
data makes those labels reportable.

**The free fix is a decision, not more labelling:** split round 2's positives,
half to training and half to a reserved rare-label test stratum. Those two
labels then get an F1 that is honest as a *conditional* number — performance
among articles the ranker surfaced — biased optimistic, and never to be
presented as a population rate. Standard practice for rare classes. It must be
decided before round 2 output is merged.

## Inter-annotator agreement: stratified, half done

Two annotators exist (HsuLinHtet 383, ZweHtetAung 85) but the 85-article overlap
was *selected* on pass 1 saying `other`, so pass 1 is constant and every kappa
is 0.000 by construction. There is no usable IAA figure yet.

Because those 85 are **all** the `other` articles, any sample blind to the second
annotator necessarily contains zero `other`. So IAA is stratified rather than
simple random — which covers the contested stratum fully instead of sampling it:

```
stratum 1  pass-1 said `other`     n= 85  weight 0.222   DONE  exact match 0.812
stratum 2  everything else         n=298  weight 0.778   TO DO 50 articles
                                                  (classifier/data/humset_iaa_task.jsonl)
overall = 0.222*0.812 + 0.778*(stratum 2)
```

At n=50 a stratum-2 result near 0.85 carries a ±10 point CI: adequate for a
course report, not for a strong claim about agreement. n=80 narrows it to ±8.

## RAG evaluation: retrieval is done, faithfulness is not

Retrieval now has real numbers and its own write-up in
`rag/eval/RETRIEVAL_RESULTS.md`. What the proposal asked for splits in two, and
only one half has been delivered.

**Done.** Recall@k on 29 known-item questions, each written from one specific
article so the target URL is known and no human judgement is needed. English
queries against Burmese articles, no translation step anywhere.

| | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| BGE-M3 dense only | 0.207 | 0.414 | 0.517 | 0.310 |
| + places gazetteer (shipped) | **0.276** | **0.483** | **0.586** | **0.391** |
| chunked, max-over-chunks | 0.103 | 0.345 | 0.483 | 0.202 |

The gazetteer gain came from a measured cause: BGE-M3 aligns *translated* names
(NUG 0.790, WFP 0.773) and fails on *transliterated* ones (Min Aung Hlaing
0.468, Myingyan→မြင်းခြံ 0.364, losing to Naypyitaw at 0.476). Nine lexical
weighting and coverage variants could not move R@10 past 0.586, and chunked
pooling moves it **down** — see §4 of that file for why the earlier score probe
predicted the opposite and was right about the mechanism anyway.

**Still owed: faithfulness.** That needs human judgement — whether each English
answer is actually supported by the Burmese sources it cites — and it cannot
start until generation exists. It does not: `ui/serve.py` retrieves and stops.
Also still owed are relevance judgements on the 15 topical questions, without
which nDCG and the true answer rate stay uncomputable.

# Phase 9 — the XLM-R arm, and the confound that nearly ruined it

```
python -m classifier.train_xlmr --probe          # tokenizer stats, no training
python -m classifier.train_xlmr                  # 5-fold CV, same folds as TF-IDF
python -m classifier.tfidf_matched_control       # TF-IDF on the text XLM-R sees
```

`train_xlmr.py` fine-tunes `xlm-roberta-base` with one head of 7 sigmoid outputs
and BCE loss — not 7 separate models, because a transformer can share a
representation across labels, which is most of the reason to expect it to help
on the rare ones. It is held to **exactly** the TF-IDF arm's protocol: same 383
articles, same 5 folds from the same seed, same label-set-signature grouping,
same `other`-exclusivity repair, same metrics, same `MIN_SUPPORT=20`. If the two
arms were measured differently the ablation would be worthless, which is the
usual way this comparison goes wrong.

Runs locally on an M2 via MPS: `torch 2.14.1`, `transformers 5.19.0`, no cost.

## XLM-R cannot read these articles, and that is a confound

```
xlm-roberta-base tokenizer on 383 Burmese articles
  chars   mean 2,109
  tokens  mean 650   median 564   p90 1,144   max 1,971
  tokens per char 0.31

at max_length=512:
  articles truncated : 230/383  (60.1%)
  tokens discarded   : 74,245 of 249,011  (29.8%)
```

Burmese is not whitespace-segmented and XLM-R's sentencepiece vocabulary was
trained mostly on other languages, so Burmese fragments heavily. **XLM-R's
position embeddings cap at 512** — this is architectural, not a setting that can
be raised.

So a straight TF-IDF-vs-XLM-R comparison would conflate two different things:
linear char n-grams versus a fine-tuned transformer, **and** the whole article
versus its first 60%. At 30% of tokens discarded, the input difference could
plausibly account for the entire gap in either direction.

## The matched-input control

`tfidf_matched_control.py` decodes each article back from the exact truncated
token ids XLM-R receives and retrains the same TF-IDF pipeline on that text.
Characters retained: **69.9%** (mean 2,109 → 1,474).

```
label                    FULL    TRUNC    delta
context                 0.787    0.810   +0.023
shock_event             0.867    0.861   -0.006
casualties              0.739    0.733   -0.006
other                   0.756    0.760   +0.004
displacement            0.627    0.472   -0.155     recall 0.565 -> 0.370
humanitarian_access     0.000    0.000    0.000     thin
information_comm        0.000    0.000    0.000     thin

macro-F1*               0.755    0.727   -0.028
exact match             0.569    0.567   -0.002
```

**Compare XLM-R against 0.727, not 0.755.** That is the matched-input number.

## Truncation damage is concentrated in one label, and length explains it

`displacement` loses 0.155 F1 while every other label moves by at most 0.023.
Two candidate explanations, and a head-vs-tail test separates them:

```
variant        context  shock_event  displacement  casualties   other   macroF1*
head 512         0.810        0.861         0.472       0.733   0.760      0.727
tail 512         0.793        0.865         0.507       0.707   0.766      0.728
FULL text        0.787        0.867         0.627       0.739   0.756      0.755
```

The intuitive story — "displacement is reported late, after the lead describes
the event" — gets only weak support: the tail beats the head by 0.035. The real
explanation is that **both halves lose badly to the whole** (0.472 and 0.507 vs
0.627), so the signal is *distributed* rather than positioned, and no 512-token
window recovers it.

Article length confirms it:

```
median chars    displacement 2,614   |   all 1,856   |   other 1,327
```

`displacement` articles are the longest in the set and lose the most to the cap;
`other` articles are the shortest (sport results are brief) and lose nothing.
Overall macro-F1 is identical for head and tail (0.727 vs 0.728), so **which**
half is kept does not matter — only how much.

Two consequences worth reporting:

- Any `displacement` weakness in the XLM-R arm is partly the input, not the
  model, because it reads the same truncated text.
- A long-context multilingual encoder is a concrete, motivated next step for
  this corpus — and "Zawgyi on/off" was always going to be a trivial ablation
  (the corpus is already Unicode), so **truncation is the ablation that actually
  has an effect** and is the better one to report.

## Run 1 was invalid, and the reason was a protocol asymmetry I introduced

The first XLM-R run scored **macro-F1\* 0.296**, and five of the seven labels
produced **zero predictions**. The tempting read is "the transformer failed on
383 articles". That read was wrong.

The logits said the signal was there. On `casualties` the mean logit for
positives sat **+0.77 above** the mean for negatives — clean separation — yet
the single highest logit in the whole column was **−0.26**. Every score was
below zero, so a `> 0` decision rule emitted nothing. The model had learned to
rank and had not learned where to cut, because with 16 positives against 367
negatives the loss is minimised by predicting "no" everywhere.

The cause was that the two arms were no longer matched. TF-IDF used
`class_weight="balanced"`; XLM-R used plain BCE with **no positive-class
weighting at all**. That is exactly the kind of quiet asymmetry the Phase 9
protocol was written to prevent, and I put it there.

Three fixes:

1. `pos_weight = n_neg / n_pos` in `BCEWithLogitsLoss`, computed **per fold from
   the training split only** and capped at 20 — the cap matters because
   `information_communication` would otherwise get a weight near 60 and dominate
   the shared representation.
2. Per-label decision thresholds chosen on the **training fold**, never the test
   fold (`train_xlmr.py:293`).
3. Epoch-**mean** loss in the log instead of the last batch, which is noise and
   cannot show overfitting.

Point 2 is the one worth dwelling on. Simply re-thresholding run 1's existing
logits **on the test fold** lifted 0.296 → 0.640, and that number is not
reportable: it is the test set choosing its own operating point. The honest
version picks thresholds on training data and pays whatever that costs.

## Results: three seeds, run on a Colab T4

```
label                        supp   s/15   s/16   s/17    mean     sd
context                       186  0.717  0.757  0.718   0.731   0.023
shock_event                   179  0.834  0.853  0.748   0.812   0.056
other                          69  0.566  0.667  0.574   0.602   0.056
casualties                     85  0.613  0.659  0.549   0.607   0.055
displacement                   46  0.358  0.379  0.331   0.356   0.024
humanitarian_access            16  0.080  0.049  0.148   0.092   0.051   thin
information_communication       6  0.000  0.040  0.000   0.013   0.023   thin

macro-F1 (support >= 20)         0.617  0.663  0.584   mean 0.621  sd 0.040
macro-F1 (all 7)                 0.452  0.486  0.438   mean 0.459  sd 0.025
micro-F1                         0.677  0.680  0.602   mean 0.653  sd 0.044
exact match                      0.399  0.352  0.282   mean 0.344  sd 0.059
Jaccard                          0.610  0.600  0.533   mean 0.581  sd 0.042
Hamming loss                     0.148  0.155  0.207   mean 0.170  sd 0.032
```

Against the matched-input baseline:

```
XLM-R     macro-F1*  0.621  (3 seeds, sd 0.040)
TF-IDF    macro-F1*  0.727  (same 512-token truncation)
                    -0.106  ->  TF-IDF wins
```

The gap is about **2.6 times the across-seed standard deviation**, so on this
dataset the linear model beating the fine-tuned transformer is a supported
result rather than a single-seed accident. The earlier single local run (0.640
on MPS, batch 8) sits inside the seed band and did not establish this on its
own.

The direction is not surprising once the sample size is in view: 383 articles
with 1.53 labels each is the regime where the Phase 7 learning curve shows
labels needing ~30 positives to reach F1 ≈ 0.5. Fine-tuning 278M parameters on
that much supervision, through a tokenizer that discards 30% of each article,
has no reason to beat char n-grams.

### The seed does not pin the result

Two of the recovered checkpoints are the **same seed** (20261015), same data,
same hyperparameters, same T4 — one from a single-seed invocation and one from
the `--seeds 3` sweep. They disagree:

```
seed 20261015, run A    macro-F1*  0.636
seed 20261015, run B    macro-F1*  0.617
max per-article score difference    0.512
```

So there are two sources of variance stacked: ~0.02 run-to-run from GPU
nondeterminism (non-deterministic kernel selection, which `torch.manual_seed`
does not control) and ~0.040 across seeds. Both are far smaller than the 0.106
gap, so the headline survives, but **any single per-label figure from this arm
carries roughly ±0.05** and should not be quoted to three decimals as though it
were stable.

A useful side effect: run A's 0.636 on a T4 against 0.640 locally on MPS means
the Colab port reproduces the Mac result, so the harness is not introducing
anything of its own.

### Where XLM-R loses, and the one place it wins

`other → context` is the single largest error on every seed — 58%, 33% and 58%
of all `other` articles respectively. That boundary has now defeated three
independent systems: a human annotator working from a vague guide (Phase 6,
pass 1 over-applied `other` to 22% of the set), a linear model (Phase 7, 27.5%
of `other` lost to `context`), and a fine-tuned transformer. It is an
intrinsically hard distinction, not a documentation failure — worth saying
plainly, because the easy conclusion after Phase 6 was that rewriting the guide
had fixed it.

XLM-R is the **only** model in either arm that scores non-zero on
`humanitarian_access` (mean 0.092 against TF-IDF's hard 0.000), and it touches
`information_communication` once (0.040 on one seed of three). This is the
shared-representation effect the single-head design was chosen for, and it is
the right shape — but at 16 and 6 positives, with across-seed sd of 0.051 on a
mean of 0.092, it is **weak evidence of a real effect, not a measurement**. The
honest framing is that it motivates Round 2's targeted collection rather than
demonstrating anything. If Round 2 lifts those two labels past ~30 positives,
this is the comparison to re-run first.

## Running the arm on Colab or Kaggle

The M2 takes ~1.6 h for one seed; a T4 takes ~10–20 min, so three seeds fit in
under an hour. `colab/xlmr_colab.ipynb` carries the full sequence and
`colab/xlmr_bundle.zip` the code and data.

```
python -m classifier.train_xlmr --probe              # verify the data came across
python -m classifier.train_xlmr --seeds 3 --batch 16 --no-grad-ckpt
```

`--probe` must report ~60% truncated and ~30% of tokens discarded before
training. If it does not, the wrong data file was uploaded.

**The bundle contains full article bodies**, so a Kaggle dataset built from it
must be set **Private** and the Colab notebook must stay in your own Drive.
`colab/*.zip` is in `.gitignore` for the same reason. Uploading is not
publishing, but it does hand a copy to a third party, and the release rule for
this project is URLs, labels, metadata and scripts only.

### Checkpointing is what makes a free runtime usable

A Colab session can disconnect at any time, and it did. `--ckpt` writes every
completed fold's **threshold-shifted scores** to JSON as soon as that fold ends
(`train_xlmr.py:194`), so a fold that finished is finished:

```
  resuming: folds [1, 2, 3, 4, 5] already done
```

Re-running against a complete checkpoint trains nothing. It loads the
tokenizer, reads the scores and prints the report — which means a run killed
at fold 5 of 5 can be finished **on a laptop with no GPU at all**, and the
three-seed table above was in fact regenerated locally from checkpoints after
the runtime that produced them was gone.

Two conditions make this work, and both are easy to get wrong:

- The checkpoint must live on **Drive**, not the runtime disk, or it dies with
  the VM. The notebook mounts Drive and writes to
  `/content/drive/MyDrive/myhumnews/`.
- The checkpoint is keyed by `{seed, n}` and refuses to resume if either
  changed, so editing the gold set invalidates it rather than silently mixing
  old and new scores.

The `--dump` output needs the same treatment, which it did not originally get:
the first version wrote the prediction dump into the runtime's working
directory, where it was lost while the checkpoints beside it survived.

# Phase 10 — how reliable are the labels the whole project rests on?

```
python -m classifier.sample_iaa --seen classifier/data/humset_iaa1_task.jsonl \
       --out classifier/data/humset_iaa2_task.jsonl
python -m classifier.score_multilabel --gold classifier/data/humset_gold.jsonl \
       --pred classifier/data/annot-iaa2/labels-ZweHtetAung.jsonl --agreement
```

Every number in Phases 7–9 is measured against `humset_gold.jsonl`, and every
one of those 383 articles was labelled by **one person, once**. If those labels
are not reproducible, macro-F1 0.727 is not a measurement of the model — it is
a measurement of how well the model imitates one annotator.

So a second annotator (ZweHtetAung) re-labelled a blind sample of gold, seeing
titles and bodies but never pass-1's labels.

## Run 1 was contaminated, by my own deployment mistake

The first 50-article pass returned exact set match **0.160**. It is not a
reliability estimate and is not reported as one.

When it was served, `HUMSET_GUIDE.md` still carried the ROUND 2 section — the
one naming `humanitarian_access` and `information_communication` as the
under-represented labels, with a table of how many more examples each needed.
The disagreement landed exactly there:

```
label                        added   dropped
humanitarian_access             23         0
information_communication        3         0
other                            7         0
context                          3        17
```

23 additions and **zero** removals on the flagged label is a priming signature,
not annotator noise. The second pass applied 2.12 labels per article against
pass 1's 1.74. The annotator was not rushing — 31 minutes for 50 articles,
median 13 s.

The run is kept (`classifier/data/annot-iaa/`, with a README recording why)
because the effect is itself a result: **one guide section naming the thin
labels shifted a careful annotator's application rate on them by roughly 8x.**
That is direct evidence about how strongly written guidance steers annotation,
and it is the same lesson as Phase 6's finding that stated LLM confidence did
not predict correctness.

## Run 2: the guide section removed, a fresh unseen sample

The ROUND 2 section was archived out of the live guide
(`HUMSET_GUIDE_round2_archive.md`), which now ends "Label what is actually in
the article. There is no quota and no label this pass is trying to collect."
`sample_iaa.py` then drew 50 articles from the **248 gold articles the second
annotator had never seen** — excluding both the 85 they re-labelled in pass 2
and run 1's 50 — stratified on label-set signature, with the best of 3,000
draws on share deviation so the label and outlet mixes match the pool.

```
                         exact match   Jaccard   mean kappa
run 1 (primed guide)           0.160     0.440        0.279
run 2 (guide stripped)         0.260     0.508        0.264
                     95% CI [0.159, 0.396]
```

```
label                        A    B   both   kappa   F1(B|A)
context                     30   30     24   0.500     0.800
shock_event                 30   18     15   0.318     0.625
casualties                  15   11     10   0.691     0.769
displacement                 7    5      2   0.245     0.333
humanitarian_access          1   15      1   0.091     0.125
information_communication    1    0      0   0.000     0.000
other                        0    8      0   0.000     0.000
```

`other` is constant in pass A **by construction** — the second annotator has
already seen all 85 articles pass 1 called `other`, because those were the
pass-2 re-labelling set, so no unseen sample can contain one. This design can
never measure `other` agreement, and `sample_iaa.py` says so in its docstring.

**Exact set match is the harshest possible multi-label metric** — one label out
of seven scores the article zero. Jaccard 0.508 and the per-label F1 column are
the fairer summary, and they tell a less alarming story than 0.260 does. Report
both.

## `humanitarian_access` is not a defined label

Removing the guide section cut the second annotator's `humanitarian_access`
rate from 52% of articles to 30%. It did not fix it:

```
base rate in the 248-article pool          4.8%
second annotator applied it to            30.0%
direction                               +14 / -0
```

Thirty percent against a 4.8% base rate, one-directional across two independent
samples, is not noise. Either the annotator is still primed by having discussed
the label at length, or — more likely, given it survived the guide fix — the two
readings of "humanitarian access" genuinely differ, one narrow (aid operations
and their obstruction) and one broad (any hardship, restriction or interference).

This matters beyond the IAA number. **Round 2 collected 101
`humanitarian_access` positives under the narrow reading**, and the second
annotator disagrees with that reading roughly 6:1. Dropping
`humanitarian_access` from the comparison lifts exact match 0.260 → 0.380, so
this single label accounts for nearly half the measured disagreement.

The fix is adjudication, not more data: the two annotators reconcile ~20 cases
together and the resulting boundary goes into the guide, before anything is
trained on those 101 positives.

## The model is at the annotation ceiling

Per-label F1 of the model against gold, beside a second human's F1 against the
same gold:

```
label              model (TF-IDF, full text)   second human
context                                0.787          0.800
casualties                             0.739          0.769
shock_event                            0.867          0.625
displacement                           0.627          0.333
```

On `context` and `casualties` the model sits **at** the human ceiling. On
`shock_event` and `displacement` it scores higher than an independent human.

This is **not** "the model beats a human", and writing it that way would be
dishonest. The model was fitted to these labels through cross-validation, so it
has learned annotator A's conventions including the idiosyncratic ones, while a
second annotator has not. The gap measures convention-learning, not competence.

What does survive: the headline is bounded by annotation reliability rather
than by model capacity, so **macro-F1 0.727 should never be quoted without an
agreement figure beside it.** Chasing a better classifier past this point
optimises agreement with one person's reading of seven labels.

## What this costs the rest of the project

- The gold set is one annotator's work. Both `casualties` (kappa 0.691) and
  `context` (kappa 0.500) reproduce acceptably; `displacement` (0.245) and
  `humanitarian_access` (0.091) do not.
- The stratified design called for
  `overall = 0.222 x 0.812 + 0.778 x (stratum 2)`, giving **0.383** with run 2.
  That number is an upper bound, not an estimate: stratum 1's 0.812 came from a
  pass where the annotator knew every article in it had been called `other`,
  which inflates agreement. Quote the stratum-2 figure instead.
- No versioned copy of the pass-1-era guide survives, so "did the instructions
  change between passes?" cannot be checked after the fact. The guide is the
  measuring instrument and it was not under version control. Copy it to a dated
  filename on every edit from here on.
- The deck carries one IAA figure, `fig12_annotation_ceiling.png`, which plots
  run 2 only and is captioned as a ceiling rather than as a model beating a
  human. **Run 1 is plotted nowhere**, because its numbers measure the priming
  and not agreement.

---

# Phase 11 — the DVB 19-label topic scheme, and a linear arm swept to exhaustion

Phase 2's 10-topic taxonomy is **deprecated**: it was designed before the
project's target was clear, and it was my invention rather than anything the
publishers use. The general-dataset scheme is now **DVB's own editorial
categories**, 19 labels that clear `MIN_SUPPORT = 100`, in
`dataset/dvb_scheme.py`. Using the publisher's categories rather than mine means
the label set needs no defending — it is what a Burmese newsroom actually files
against.

```
.venv/bin/python dataset/tune_dvb.py      # C / max_features / min_df + thresholds
.venv/bin/python dataset/sweep_dvb.py     # features and model classes
```

Protocol, and every later number inherits it unchanged: **temporal split**,
train on articles published ≤2024, test on 2025+, with validation carved out of
train at 15%. Per-label decision thresholds are fitted on validation and scored
on test, never fitted on test.

```
articles 62,016    train 31,855   val 5,622   test 24,539
```

## The arm is exhausted, and only one knob ever mattered

```
baseline, threshold at 0                    0.644
+ per-label thresholds                      0.658   (+0.014)
+ tuned C / max_features / min_df           0.658   (+0.000)
```

Thresholds bought 0.014. **Nothing else bought anything.** C across
{0.25, 0.5, 1, 2}, max_features across {300k, 600k}, min_df across {2, 3},
max_df, and the n-gram range all land on the same number. The shipped config is
`C=0.5, max_features=300_000, min_df=2`, in `dataset/dvb_model_config.json`.

Two things I asserted during this and got wrong, recorded because the
corrections are the useful part:

- I called `class_weight='balanced'` the biggest untested knob. It was already
  in the code, along with `sublinear_tf=True`. Read before claiming.
- I expected the `char` analyzer might beat `char_wb`. Measured: 0.653 vs
  0.658.

## Four model classes, and the one that should have won did not

```
LinearSVC            0.6584   <- shipped
ClassifierChain      0.6456
LogisticRegression   0.6353
ComplementNB         0.2206
```

`ClassifierChain` was the interesting candidate: it feeds each label's
prediction forward as a feature, ordered here by descending F1 so strong labels
inform weak ones, which is exactly the structure a 19-label scheme with heavy
co-occurrence should reward. It lost by 0.013. I had called it "the likeliest
source of any real gain" beforehand; it was not. `ComplementNB` collapses to 0.22 and no
cause was isolated — it was swept for completeness, not because it was
expected to compete.

## The matched-input control, and why 0.658 is the wrong target

XLM-R cannot read past 512 tokens. Burmese fragments at **0.306 tokens per
character** under XLM-R's tokenizer, so the mean article is 586 tokens against
that 512 ceiling. Comparing a transformer at 512 tokens against a linear model
reading whole articles confounds "which model is better" with "which model saw
more text" — the same confound that nearly ruined Phase 9.

`dataset/train_dvb_xlmr.py --linear-control` truncates every article to its
first 512 XLM-R tokens, decodes it back, and runs the identical linear arm on
exactly that:

```
linear arm, whole articles                  0.658
linear arm, first 512 XLM-R tokens          0.643   <- the honest target
cost of truncation, charged to neither      0.015
characters surviving truncation             74.4%
```

Truncation costs only 0.015 because Burmese news signal is front-loaded in the
headline and lede. **0.643 is what the transformer must beat.** Both figures
ship in `dataset/dvb_xlmr_meta.json` as
`f1_per_label_linear_at512` and `f1_per_label_linear_full_text` — named
separately after I spent an afternoon comparing per label against the wrong one
and understating the transformer on every label where truncation had hurt the
baseline.

---

# Phase 12 — the DVB XLM-R arm: the first thing that moved the number

```
.venv/bin/python dataset/export_dvb_xlmr.py        # flat file for a free GPU
dataset/dvb_xlmr_colab.ipynb                       # the run
dataset/dvb_xlmr_sweep_kaggle.ipynb                # the hyperparameter sweep
```

The split, the label set and the validation carve-out are resolved **locally**
and baked into `dataset/dvb_xlmr_data.jsonl.gz` as a `split` field, so the
notebook holds no project logic that could drift from the protocol those
baselines were measured under. 62,016 rows, 46 MB.

## Why a free GPU, and what it actually cost

Measured on this M2 via MPS before moving: 512 tokens with gradient
checkpointing ran **4.82 s/step, 6.2 h/epoch** — about 19 hours for three
epochs. A Colab T4 does the same epoch in ~2 hours. Gradient checkpointing was
*faster* locally only because 16 GB of unified memory was under pressure; on a
16 GB T4 with fp16 the model needs about 8 GB, so there it is pure overhead,
roughly 1.4x on step time.

## Result, all 3 epochs

```
Tesla T4, lr 2e-5, batch 16, 512 tokens, pos_weight capped at 50
macro-F1 by epoch   0.6353 -> 0.6599 -> 0.6718
vs linear @512 (0.643)                   +0.029
vs linear full text (0.658)              +0.014
excluding COVID19 (18 labels)            +0.069
weighted-F1                               0.821   (linear 0.799)
labels won                               16 of 19
```

The run took three Colab sessions: the free GPU quota expired after epoch 2 and
again before epoch 3 finished. Per-epoch checkpointing is the only reason this
cost nothing — each session resumed from the last finished epoch.

**Epoch 3 was worth +0.012**, which is the OneCycleLR anneal landing: at the
end of epoch 2 the learning rate is still near 6e-6, and epoch 3 decays it to
zero. Stopping at epoch 2 would have reported a model caught mid-schedule.

Report this as **"three epochs, final epoch reported"** and not as "best
epoch". The curve is monotone, so the last epoch is also the best one and no
selection was required — but the notebook revision used for this run chose its
reported epoch on the TEST score, so calling it "best" would describe a
selection that should never have been available. `report_dvb_xlmr.py` prints a
warning when it reads a config from that revision.

**0.7022 excluding COVID19 is close to the ceiling.** Phase 13's label-
consistency work puts the achievable macro around 0.73, given that four labels
are applied inconsistently by the publisher. The remaining headroom on this
scheme is mostly annotation quality, not modelling.

## Where it wins, and the pattern that explains why

```
label                             lin@512    xlmr    delta
သဘာဝပတ်ဝန်းကျင် (environment)      0.691   0.859   +0.168
အနုပညာ (art)                       0.538   0.663   +0.125
ပညာရေး (education)                 0.605   0.726   +0.121
သတင်းမီဒီယာ (news media)           0.565   0.683   +0.117
အမျိုးသမီး/လူငယ် (women/youth)     0.337   0.451   +0.114
ရွေးကောက်ပွဲ (election)            0.646   0.745   +0.099
တိုင်းရင်းသား (ethnic)             0.249   0.345   +0.096
...
နိုင်ငံရေး (politics)              0.580   0.565   -0.015
အလုပ်သမား (labour)                 0.792   0.765   -0.028
ဘာသာရေး (religion)                 0.424   0.372   -0.052
COVID19                            0.818   0.133   -0.685
```

**The gains concentrate where training data is scarce**, which is the whole
reason to expect pretraining to help:

```
train support < 1000    n=10   mean delta +0.069
train support >= 1000   n= 8   mean delta +0.040
corr(train support, gain) = -0.448   (n=18, excl COVID19)
```

Women/youth has **76 training examples** and gains the most in relative terms.
Election has 96 training positives against 458 in test, because the 2025
election falls entirely after the temporal split — char n-grams had nothing to
generalise from, and pretrained multilingual weights did. Treat −0.448 as
directional at n=18; the group split is the firmer evidence.

## The one catastrophic label, and why it is not a model failure

COVID19 collapses from 0.818 to 0.133 at precision 0.049 / recall 0.909 — the
model fires on nearly everything. The cause is **temporal prevalence shift**,
not learning:

```
COVID19 positives    train 1,282    val 202    test 11
```

A threshold fitted on a validation slice where COVID is 3.6% of articles cannot
work on a test period where it is 0.045%. The linear arm survives it better
only because LinearSVC margins are less confident than a fine-tuned sigmoid, so
they degrade more gracefully. That is luck, not capability. The label costs
0.029 of macro-F1 on its own: had it scored like the mean of the other 18
(0.689) instead of 0.133, macro would be 0.689 rather than 0.660.

## Two protocol fixes made to the notebook

- **Epoch selection was on test.** `if macro > best[0]` picked the best epoch by
  its test score. It now computes validation macro-F1 at the same per-label
  thresholds and selects on that, reporting test alongside. Harmless on a
  monotone 2-epoch run; fatal the moment a sweep makes epochs non-monotone.
- **Only macro-F1 was reported.** It now emits macro, micro and weighted with
  micro precision/recall. See Phase 13 for why that matters more than it sounds.

## Checkpointing is what makes a free runtime usable

Three epochs need ~6 hours and Colab free will not give you that uninterrupted.
After every epoch the notebook writes a results JSON and a full checkpoint
(model + optimizer + scheduler, 3.3 GB, or 1.1 GB with
`SAVE_OPTIMIZER = False`) to Drive via an atomic `.tmp` + `os.replace`, and on
restart resumes from the last finished epoch. It also locates its own data —
seven exact paths, then globs, then a recursive Drive scan — so there is no
upload step to repeat after every disconnect.

## The sweep, and the asymmetry it exists to fix

The linear arm got eight knobs swept. XLM-R ran **once**, at library defaults,
and still won 15 of 19 labels. `dataset/dvb_xlmr_sweep_kaggle.ipynb` closes
that gap inside Kaggle's ~30 GPU-hours/week:

| phase | configs | data | epochs | eval | cost |
|---|---|---|---|---|---|
| A — learning rate | 4 | 40% | 2 | val | ~5.2 h |
| B — pos_weight cap | 2 | 40% | 2 | val | ~2.6 h |
| C — final, full data | 1 | 100% | 3 | val + test once | ~4.8 h |

It **never loads the test split** — not "does not score it", does not read it
off disk — so selection cannot leak even by accident. Finished configs append
to `sweep.json` and are skipped on restart, because a 9–12 hour session will not
survive the whole sweep. No weights are saved during the search: six 3 GB
checkpoints do not fit in a 20 GB working directory, and re-running one config
is cheaper than storing them all.

Phase B moves the `pos_weight` cap because precision 0.049 at recall 0.909 is
what over-weighting a rare label looks like. If the winner beats the defaults by
under ~0.005, the honest conclusion is that the defaults were already right —
which still answers the question, by converting "untuned" into "tuned, and
tuning was not the explanation".

---

# Phase 13 — are these numbers actually weak? Three measurements say no

A macro-F1 of 0.66 reads low beside published multi-label work. Three
diagnostics were run to find out how much of that is the model and how much is
the metric, the protocol and the annotations. The answer is: very little of it
is the model.

## 1. The metric. Papers headline micro-F1; macro was being quoted against it

```
.venv/bin/python dataset/split_protocol_cost.py
```

Same model, same labels, same predictions, three aggregations:

```
macro-F1     0.6584   over 19 labels
micro-F1     0.8043   (P 0.822 / R 0.788)
weighted-F1  0.7994   over 27,596 test assignments
```

For reference, a published baseline reports micro-F1 **0.753 on AAPD** (55,840
documents, 54 labels) and **0.896 on RCV1-v2** (804,414 documents, 103 labels),
both English, both random splits. **Micro-F1 0.804 on 32k Burmese training
articles under a temporal split sits between them.**

Macro gives a label with 76 training examples the same weight as one with
9,426, so on a long-tailed set it reads ~0.15 lower on identical predictions.
Report all three; the macro-micro gap *is* the long-tail finding.

## 2. The protocol. The temporal split costs 0.044 macro and ~nothing on micro

```
                temporal   random    cost
macro-F1         0.6584    0.7024   +0.0440
weighted-F1      0.7994    0.8131   +0.0137
micro-F1         0.8043    0.8071   +0.0028
```

The random split is a **diagnostic only** and must never be quoted as a result
— choosing the easier protocol after seeing both is exactly the move this file
exists to prevent. The temporal figure stays the headline, because predicting
next year's news from this year's is what a deployed classifier faces.

I expected the cost to fall on rare labels. It does not — grouped by support it
is *larger* for common ones (+0.067 vs +0.024). What moves is **time-bound
topics**, regardless of size: election +0.189, COVID19 +0.149, social +0.129,
environment +0.100. Four labels do *better* under the temporal split. Micro
barely moves because international (30% of all test assignments) and sport
(11%) sit at ~0.98 under both protocols — 41% of micro's mass is temporally
stable.

And part of that +0.044 is not difficulty but **leakage**: news corpora cover
each event many times, so a random split puts near-duplicate articles about one
event on both sides of the train/test line. 0.044 is an upper bound on the
genuine cost.

## 3. The annotations. Label consistency predicts F1 at r = +0.874

```
.venv/bin/python dataset/label_consistency.py --thresholds 0.3 0.35 0.4 0.5 0.6 0.7
```

The DVB labels are the publisher's own editorial tags, so there is no second
annotator and no inter-annotator agreement to compute. Consistency can still be
measured without one: a news corpus covers each event many times, so if two
near-identical articles about the same event get different labels, the label is
being applied inconsistently.

Articles are paired when published within ±3 days with TF-IDF cosine above a
threshold, comparing only the first 1,200 characters — a news event is its
headline and lede, and later paragraphs diverge between follow-ups. Then per
label, over pairs where at least one article carries it,
`agreement = |both| / |either|`.

```
cosine >= 0.35,  13,283 pairs,  all 19 labels
label                            prev    agree      F1
sport                            8.6%    0.986   0.985
နိုင်ငံတကာ (international)      23.9%    0.965   0.980
COVID19                          2.4%    0.966   0.750
သဘာဝပတ်ဝန်းကျင် (environment)   4.0%    0.895   0.732
နိုင်ငံရေး (politics)           19.8%    0.564   0.599
ကျန်းမာရေး (health)              2.4%    0.403   0.745
ဘာသာရေး (religion)              0.3%    0.286   0.370
အမျိုးသမီး/လူငယ် (women/youth)  0.3%    0.250   0.250
တိုင်းရင်းသား (ethnic)           1.7%    0.178   0.264
လူမှုရေး (social)                3.9%    0.152   0.285

corr(agreement, F1) = +0.874
```

**The four lowest-agreement labels are exactly the four lowest-F1 labels.**
Agreement rises with prevalence by chance alone — `p/(2-p)` for a label on
fraction `p` — so the script reports that chance level per label, and the
comparisons to lean on are prevalence-matched:

```
environment   4.0% prevalence   agree 0.895   F1 0.732
social        3.9% prevalence   agree 0.152   F1 0.285
                       0.02 pts apart, F1 gap 0.447

education     1.6% prevalence   agree 0.691   F1 0.708
ethnic        1.7% prevalence   agree 0.178   F1 0.264
                       0.08 pts apart, F1 gap 0.444
```

Social at 0.152 means that when two near-identical articles cover the same
social story, **only 15% of the time do both carry the tag**. So social and
ethnic are not hard labels; they are inconsistently applied labels, and the
classifier reproduces the inconsistency to within 0.01 of F1.

Two informative outliers separate two different failure modes. **COVID19 has
0.966 agreement and 0.750 F1** — annotated very consistently, failing purely on
the prevalence collapse. **Health has 0.403 agreement and 0.745 F1** —
inconsistently tagged but still classifiable, because health vocabulary is
distinctive even when the editor forgot the tag.

Limits: the correlation weakens at stricter thresholds (+0.58 at 0.6, +0.31 at
0.7) purely from range restriction, since only 6 labels clear the 20-pair
minimum there and all are high-F1. Women/youth rests on 20 pairs. Use 0.35 as
the operating point — the only setting covering all 19 labels, and the highest
correlation.

## Two label-set changes considered and rejected

Both would have raised macro-F1. Both are recorded here with the numbers that
killed them, because "we considered this and did not do it" is only credible
with the measurement attached.

**Merging COVID19 into health.** Justified independently of any test score:
DVB's editors co-assign `health` to 46.2% of COVID19 articles, another 48.2%
carry COVID19 as their only label, and just 5.6% carry a competing topic. Since
the scheme is multi-label, merging *adds* health and drops the COVID19 column,
so nothing is relabelled away from a category it was in. Measured with
`dataset/merge_covid_health.py`:

```
linear, full text    19 labels 0.6584  ->  18 merged 0.6513   (-0.0071)
linear, @512 tokens  19 labels 0.6430  ->  18 merged 0.6335   (-0.0095)
```

It *hurts* the linear arm, because COVID19's 0.818 is above its macro mean. But
it would roughly **triple the headline XLM-R delta**, from +0.017 to about
+0.05, since it deletes the single label where the baseline wins by 0.685. Not
shipped. The dataset keeps 19 labels and the write-up reports the 19-label
comparison, with the merge available as a secondary analysis provided that
swing is stated.

**Dropping the floor labels.** Worth +0.067 to +0.075 of macro. Rejected for
three reasons: the worst three are not the same three across models (the
baseline's include women/youth, which is the transformer's biggest win); a fixed
set *shrinks* the headline delta from +0.0168 to +0.0140, because XLM-R's gains
live in those labels; and `ethnic` and `social` are among the most
humanitarian-relevant categories in a Myanmar corpus. The consistency
measurement above is what replaced it — the floor is now explained rather than
hidden.

## What to report

Macro, micro and weighted together. The temporal-split cost as measured, with
the random figure labelled a diagnostic. All 19 labels, with
`corr(agreement, F1) = +0.874` as the reason four sit at the floor and the
implied ceiling near 0.73. COVID19 footnoted as unmeasurable in the 2025 test
period rather than quietly removed.

```
dataset/dvb_merge_covid_health.json     the merge, measured and declined
dataset/dvb_split_protocol_cost.json    temporal vs random, all three metrics
dataset/dvb_label_consistency.json      agreement vs F1, 6 thresholds, 19 labels
```
