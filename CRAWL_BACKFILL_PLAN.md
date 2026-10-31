# Backfill plan — extending the corpus to February 2021

**Date:** 2026-10-17
**Why:** the corpus starts 2025-01-01. The coup was 2021-02-01. The four missing
years hold the densest humanitarian reporting in modern Burmese news — the
Sagaing offensives, Cyclone Mocha, the 2023 Rakhine fighting — and none of it is
in the dataset.

**What this does not do:** it does not help the classifier. The bottleneck there
is labels (743 of 38,152). More raw articles makes the labelled fraction
smaller. This is for the **dataset deliverable** and the **RAG retrieval pool**.

---

## Current state, measured

```
corpus          38,152 articles   2025-01-01 .. 2026-09
                1.5 GB on disk    (raw gzipped HTML is 82% of it)
dvb 26,119 · irrawaddy 4,390 · bbc 2,761 · myanmarnow 2,454 · rfa 2,428
```

VM `azureuser@70.153.9.116`: **29 GB disk, 21 GB free**, 842 MB RAM, 4 GB swap.
The crawler is **not deployed** there — only `~/label` (the annotation service)
and an unrelated `~/sentinel`. No corpus, no SQLite queue.

---

## Reachability — probed, not assumed

| outlet | mechanism | 2021 reachable? | evidence |
|---|---|---|---|
| RFA | daily sitemaps | **yes** | 12–23 articles/day for 2021-02-05, 2021-03-15, 2022-06-10, 2023-05-15, 2024-02-01 |
| Myanmar Now | WP REST API | **yes** | 146 posts in Feb 2021, 206 in Jun 2022, 188 in May 2023 (vs 105 in Jan 2025) |
| DVB | category pagination | **yes** | page 1 = Oct 2026, 900 = Jul 2024, 1500 = Mar 2023, **2200 = Dec 2021**; 40 items/page |
| BBC | Wayback CDX | **NO** — see below | `/burmese/articles*` has **0** captures in 2021, 7 in 2024 |
| Irrawaddy | day-indexed | yes | unchanged mechanism |

### Estimated yield (extrapolated from 21 months of actuals — estimates, not counts)

```
outlet         now      added (est)      total (est)
dvb         26,119       ~58,000           ~84,000
rfa          2,428       ~18,000           ~20,000
bbc          2,761       ~28,000*          ~31,000     * needs code, see below
irrawaddy    4,390        ~9,800           ~14,000
myanmarnow   2,454        ~8,500           ~11,000
                        ~105,000          ~143,000
```

Disk at 143,000 articles: **~5.6 GB**. Fits in 21 GB.

RFA's figure is the softest: those sitemaps include interviews and programme
pages, and the crawler filters by URL pattern. ~18,000 assumes ~65% yield.

For scale, the largest Burmese news dataset on HuggingFace is ~29k rows.

---

## The blocker nobody would notice: saved discovery state

**`--start 2021-02-01` alone discovers almost nothing.** Each adapter persists
resume state that assumes the Jan 2025 window, and it silently short-circuits:

| outlet | stored state | effect of a naive 2021 run |
|---|---|---|
| RFA | `day_cursor = "2026-09-26"` | `if cursor and day <= cursor: continue` skips **every** day 2021→2026. Discovers **zero**. |
| BBC | `cdx_done = true`, 26 topics `exhausted:true` | `if ctx.state.get("cdx_done"): return` — CDX skipped entirely. |
| Myanmar Now | `api_page = 25`, `api_seen = 2455` | starts at page 26 of a much larger result set; skips the newest ~2,500 posts of the 2021+ window. |
| DVB | `categories` table (50 rows, "paged past 2025-01-01 at page N") | written by the **old standalone** `dvb_crawler.py`; the current engine reads `discovery_state`, finds none, and restarts at page 1. Costs ~470 wasted listing pages (~16 min at `Crawl-delay 2`), fetches **no** duplicate articles. |

A naive run would look successful and produce nothing. Resets required before
starting.

---

## Deployment

The five `crawl.db` files are what prevent re-fetching the 38,152 articles
already held. Verified: the `urls` table schema in `data-vm/dvb/crawl.db`
matches `crawler/core.py` (`post_id INTEGER` vs `TEXT` is harmless — SQLite is
dynamically typed).

```
dvb         26,114 done   5.1 MB
irrawaddy    4,390 done   1.1 MB
bbc          2,761 done   912 KB
rfa          2,428 done   840 KB
myanmarnow   2,454 done   588 KB
                          8.4 MB total
```

Steps:

1. rsync `crawler/`, `run.py`, `verify.py`, `requirements` to the VM
2. create `~/MyHumNews/data/<outlet>/` and rsync each `crawl.db` into place
3. apply the state resets above (one `UPDATE`/`DELETE` on `discovery_state` per outlet)
4. `python run.py <site> run --start 2021-02-01 --end 2024-12-31`
5. verify, then rsync the new data down and merge into `data-vm/`

`--end 2024-12-31` bounds the window so nothing already held is re-examined.
**Caveat for DVB:** `--end` is enforced at parse time (`core.py:463`), not at
discovery, so DVB would still *fetch* 2025–26 articles before discarding them —
except that the uploaded `crawl.db` already marks them `done`, so they are
skipped at the queue. This is why step 2 must precede step 4.

---

## Order of work

**1. RFA + Myanmar Now** — state reset only, sitemap/API driven, exact counts,
a few hours each. Run these first: they confirm or refute the ~18,000 and
~8,500 estimates before committing the VM to a 30-hour crawl.

**2. DVB** — no reset needed, ~24–36 hours at `Crawl-delay 2`. Resumable and
keyed by URL, so disconnects cost nothing. Biggest single gain (~58,000).

**3. Irrawaddy** — **Mac only.** Cloudflare blocks Azure IPs. Run locally,
rsync up (or just merge locally).

**4. BBC** — 912 recovered; the rest blocked on Internet Archive.

**CORRECTION (2026-10-17).** My first assessment queried CDX with
`from=YYYY0101&to=YYYY1231`, which filters on **when Wayback archived** a page,
not when it was published. `bbc.py`'s own comment warns about this. So two
earlier claims were measuring the wrong thing and are withdrawn:

- ~~"`/burmese/articles*` has 0 captures in 2021"~~
- ~~"~28,000 URLs available across 2021-24"~~

What is actually established:

- `/burmese/articles/<cps_id>` exists back to **2022-07**, proven by 912 URLs
  already in `data-vm/bbc/crawl.db` with `status='skipped'` (fetched under
  `--start 2025-01-01`, then rejected by the date filter).
- **912 articles recovered with zero network requests** by re-parsing the
  stored raw HTML (`.scratch/vm/reparse_bbc.py` -> `data-backfill/bbc/`).
  2022: 48 · 2023: 290 · 2024: 574. Median body 3,445 chars, 0 parse failures,
  0 bodies under 200 chars, 0 overlap with the existing corpus.
  This works because the engine writes the raw layer *before* parsing.
- **`parse()` needs no work** — now verified on 912 real pages plus a 2014
  `.shtml` URL. BBC re-renders old articles through the current Next.js stack,
  so `__NEXT_DATA__` is present throughout.
- Fetching works from Azure (HTTP 200, 267 KB, `__NEXT_DATA__` present).
  CDX is needed once, for discovery only.
- The live `https://www.bbc.com/burmese/sitemap.xml` is NOT an alternative
  index: 100 URLs, all September 2014.

Remaining, both needing Internet Archive (currently **"Temporarily Offline"**
site-wide, not rate-limited):

1. enumerate any further `/articles/` URLs for 2022-07..2024 by clearing
   `cdx_done` and re-running discovery
2. determine the pre-2022-07 URL shape, then extend the CDX query and
   `ARTICLE_RE`. **Do not write these patterns until a real legacy URL has been
   observed** — the shape is currently unknown, not merely unconfirmed.

---

## Open questions

- **Titles in the release.** The release rule is "URLs, labels, metadata and
  scripts". Headlines are short article text sitting on that line. Including
  them makes the dataset inspectable without re-crawling; excluding them is the
  conservative reading. Undecided.
- **Does the date range change the label distribution?** 2021–22 is more
  conflict-dense than 2025–26, so `shock_event`, `casualties` and
  `displacement` will likely be commoner. The 383-article gold set was sampled
  from the 2025+ window only, so **gold stops being population-shaped for the
  enlarged corpus.** This matters for any claim that CV scores predict corpus
  performance, and would argue for sampling a fresh gold slice from the
  backfilled years.

---

## Resume point — 2026-10-17 23:05 (Irrawaddy stopped, low battery)

Irrawaddy was stopped deliberately with SIGTERM (PID 77950), not crashed.
`pragma integrity_check` on `data/irrawaddy/crawl.db` returns `ok`, and
`parsed/articles.jsonl` has exactly 11,418 lines, matching the `done`
count — no torn record.

```
done     11418
pending   3151
failed      16
```

Discovery is **finished**: all 68 `post-sitemap*.xml` are in both
`post_sitemaps` and `sitemaps_done`, with `in_range_count = 10180`.
So resuming drains the pending queue only; it will not re-discover.

Resume with the same command, from this directory, on the Mac:

```
.venv/bin/python run.py irrawaddy run --start 2021-02-01 --end 2024-12-31
```

At the observed ~30 articles/min that is about 1h45m for the remaining
3,151. Irrawaddy has to run on the Mac — it blocks the VM's IP.

The 16 failures are worth a look before the next run, in case they are a
single recoverable cause rather than 16 dead URLs:

```
sqlite3 data/irrawaddy/crawl.db "select url from urls where status='failed';"
```

VM crawls were left running and are unaffected: dvb 26,558/78,407 (still
discovering), rfa 8,256/23,536, myanmarnow 9,587/11,440.
