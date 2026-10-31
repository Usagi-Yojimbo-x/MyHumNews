#!/usr/bin/env python3
"""DVB's own editorial categories as the label scheme for the general dataset.

Replaces classifier/taxonomy.py, which mapped five outlets onto 10 invented
classes. These labels are DVB's real desks, so nothing here is my invention
-- only the merges and drops below, each with a reason.

Plan: train on DVB's 78,083 labelled articles, predict onto the other four
outlets (RFA has no topical categories at all, Myanmar Now almost none).
"""

# ---------------------------------------------------------------- merges
# Same desk under two names. DVB dropped the "/drugs" half during 2026:
# မှုခင်း appears ONLY in 2026 (338), မှုခင်း/မူးယစ်ဆေး runs 2021-2026 and
# falls off in 2026 (1,089 -> 511). Exactly 1 article of 5,132 carries both.
# Checked the text too: the "crime/drugs" sample was sexual exploitation by
# prison officials, no drugs in it. Keeping them apart would teach the
# classifier what year an article is from, not what it is about.
MERGE = {
    "မှုခင်း":              "crime",
    "မှုခင်း/မူးယစ်ဆေး":     "crime",
    # Football is a sub-tag DVB applies to nearly every sport story: 5,105 of
    # 5,332 sport articles also carry ဘောလုံး, only 17 are football-only.
    # ၂၀၂၆ ကမ္ဘာ့ဖလား is one tournament inside that. Three names, one desk.
    "အားကစား":              "sport",
    "ဘောလုံး":              "sport",
    "၂၀၂၆ ကမ္ဘာ့ဖလား":       "sport",
    # Long-tail strays that plainly belong to a desk we already have.
    # ~50 articles in total, so this is for consistency, not coverage:
    # leaving a bare "drugs" label next to a merged crime desk, or a tennis
    # label next to a merged sport desk, would contradict the scheme.
    "မူးယစ်ဆေးဝါး":          "crime",
    "တင်းနစ်":               "sport",
    "ONE CHAMPIONSHIP":      "sport",
    "၂၀၂၂ ကမ္ဘာ့ဖလား":       "sport",
}

# Labels under this many articles are dropped from the training set: the
# tail runs from 32 articles down to 1 ("Any Questions?", "ရေဒီယို"), which
# cannot be learned and cannot be evaluated.
MIN_SUPPORT = 100

# ---------------------------------------------------------------- drops
# Generic: on most articles, carries no topic.
DROP_GENERIC = {"မြန်မာသတင်း", "Lead Story"}
# Format/placement, not subject matter.
DROP_FORMAT = {"ဆောင်းပါး", "ရုပ်သံ", "သတင်းဓာတ်ပုံ", "ဓာတ်ပုံ", "ကာတွန်း",
               "မေးမြန်းခန်း", "သုံးသပ်ချက်", "သရော်စာ", "Data Story"}
DROP = DROP_GENERIC | DROP_FORMAT

# NOTE sport is effectively a 2023+ label: 1 article in 2021, 0 in 2022.
# A model trained on it partly learns publication date. Same shape for
# COVID19 in reverse (1,100 in 2021 -> 1 in 2026), though that one is a
# real-world trend rather than a scheme change.

# ------------------------------------------------------------- kept, noted
# နိုင်ငံတကာ (14,813, the second-largest label) is DVB's FOREIGN DESK, not a
# subject: it means "not Myanmar". Six sampled at random were a war in Gaza,
# a war in Ukraine, a Thai election, a Spanish protest, North Korean
# diplomacy and a Norwegian court case -- every one had a real topic and
# none carried a topical tag.
#
# Kept deliberately (Zwe, 2026-10-20) because it is authentically DVB's own
# scheme. The consequence to report, not hide: it absorbs foreign conflict
# and politics stories, so those classes are trained on Myanmar-only
# examples, and a classifier will not transfer this label to outlets that
# do not organise by desk.
OPEN = []


# NOTE: an earlier version collapsed every article to ONE label via a
# resolution order. That was wrong -- it also collapsed articles carrying
# two genuine desks (peace/war + politics became peace/war alone, costing
# politics 4,010 articles). Only the GENERIC and FORMAT tags are dropped;
# two real desks on one article are both kept. The scheme is multi-label.


def labels(categories) -> list[str]:
    """DVB categories -> every surviving label (multi-label view)."""
    out = []
    for c in categories or []:
        c = (c or "").strip()
        if c in DROP:
            continue
        out.append(MERGE.get(c, c))
    return sorted(set(out))


# ------------------------------------------------------- per-label quality
# THE DATASET SHIPS ALL 19 LABELS. This is a general-purpose news dataset,
# so breadth matters more than macro-F1: a news corpus without politics,
# sport or international is not a general news corpus. Users filter by
# quality themselves, which is why the F1 for every label is published
# rather than used to prune the scheme.
#
# Measured on DVB with a TEMPORAL split (train <=2024, test 2025+), which
# is harsher than random and the right number for a transfer task.
F1_TEMPORAL = {
    "sport":                            0.985,
    "နိုင်ငံတကာ":                        0.980,   # international
    "ငြိမ်းချမ်းရေး/စစ်ရေး":             0.824,   # peace / war
    "အလုပ်သမား/ရွှေ့ပြောင်းလုပ်သား":      0.798,   # labour
    "COVID19":                          0.750,   # only 11 test articles
    "ကျန်းမာရေး":                        0.745,   # health
    "စီးပွားရေး":                        0.741,   # economy
    "စိုက်ပျိုးရေး/တောင်သူလယ်သမား":       0.735,   # agriculture
    "သဘာဝပတ်ဝန်းကျင်/ရာသီဥတု/မိုးလေဝသ":   0.732,   # environment
    "crime":                            0.728,
    "ပညာရေး/နည်းပညာ":                    0.708,   # education / tech
    "အနုပညာ":                            0.684,   # arts
    "ရွေးကောက်ပွဲ":                       0.627,   # election
    "နိုင်ငံရေး":                         0.599,   # politics  (n=4,189)
    "သတင်းမီဒီယာ":                       0.588,   # news media
    "ဘာသာရေး":                           0.370,   # religion
    "လူမှုရေး":                           0.285,   # social   (n=899, catch-all)
    "တိုင်းရင်းသား":                      0.264,   # ethnic
    "အမျိုးသမီး/လူငယ်":                   0.250,   # women / youth
}

# A high-confidence subset for anyone who wants precision over coverage.
# NOT the scheme -- just the 8 labels that cleared F1 0.70 on the temporal
# split. Trained alone they reach macro-F1 0.843 over 29,624 articles,
# against 0.652 over 62,016 for the full 19. Note the trade is partly an
# easier task, not only a better model: removing politics, social and
# international also removed the labels that overlapped everything else.
HIGH_CONFIDENCE = [
    "ငြိမ်းချမ်းရေး/စစ်ရေး", "အလုပ်သမား/ရွှေ့ပြောင်းလုပ်သား", "ကျန်းမာရေး",
    "စီးပွားရေး", "စိုက်ပျိုးရေး/တောင်သူလယ်သမား",
    "သဘာဝပတ်ဝန်းကျင်/ရာသီဥတု/မိုးလေဝသ", "crime", "ပညာရေး/နည်းပညာ",
]
