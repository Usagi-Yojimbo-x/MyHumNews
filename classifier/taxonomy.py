"""
Topic taxonomy for the MyHumNews corpus.

The five outlets carry 416 distinct labels between them and only 16 appear in
more than one outlet, so the outlets' own labels cannot serve as a shared
target. This module maps them onto 10 coarse topic classes.

The labels are a mix of five different things:

    real topics      ငြိမ်းချမ်းရေး/စစ်ရေး, စီးပွားရေး, အားကစား     -> mapped
    format/placement Lead Story, Video, ဆောင်းပါး, Cartoon         -> DROP
    generic "news"   သတင်း, သတင်းများ, မြန်မာသတင်း                  -> DROP
    geography        အမေရိကန်, တရုတ်, ရန်ကုန်                        -> foreign desk
    named entities   မင်းအောင်လှိုင်, အာရက္ခတပ်တော်                  -> via their domain

DROPPED labels are not errors: an article labelled only "Video" or only
"သတင်း" carries no topical signal, so it contributes no usable training
example. Articles that map to nothing are excluded from the dataset.

Articles usually carry several labels, so RESOLUTION_ORDER breaks ties: the
most specific topic wins. An article tagged both ငြိမ်းချမ်းရေး/စစ်ရေး and
နိုင်ငံရေး is conflict, not politics, because the conflict framing is the more
specific claim.

!! REVIEW REQUIRED !!
This mapping was built by reading the Burmese label strings. It is a first
draft and the single highest-leverage thing to check in this project -- every
downstream metric inherits its errors, and a mislabelled class is invisible in
the accuracy number. Read CLASSES and MAPPING below and correct anything that
does not match how you would classify these outlets' sections.
"""

from __future__ import annotations

CLASSES = [
    "conflict",       # armed conflict, military, armed groups, peace process
    "politics",       # governance, elections, parliament, law, political figures
    "crime",          # crime, drugs, trafficking, scam operations
    "economy",        # business, trade, agriculture, labour, energy
    "environment",    # weather, climate, earthquakes, disasters
    "health",         # health, medicine, disease
    "education",      # education, science, technology
    # NOTE: rights/society were tried as separate classes and removed. As their
    # own class they scored F1 0.274 (lumped) and 0.299/0.222 (split), bleeding
    # ~25% into politics either way -- they carry no distinct lexical signature.
    # Folding them into politics raised macro-F1 from 0.748 to 0.796 while
    # keeping every article. Dropping them instead scored higher still (0.816)
    # but only by deleting 977 hard examples, which measures an easier problem.
    "sport",          # sports
    "culture",        # art, culture, religion, entertainment, lifestyle
    # !! `international` IS NOT A TOPIC -- it is the outlets' foreign DESK.
    # Measured, not guessed: on 200 hand-labelled DVB articles it accounted for
    # 46% of all outlet/hand disagreements, and 0 of those 32 articles
    # mentioned Myanmar. DVB files a Venezuelan earthquake, Russian strikes on
    # Ukraine and an Australian bird flu outbreak all under it, while a human
    # reading them says environment, conflict and health. It records WHERE a
    # story was placed, not what it is about.
    #
    # It stays because all 8,674 articles carrying it carry NO other mappable
    # label, so dropping it would delete 29.7% of the dataset. The name was
    # kept for continuity. But it should be reported separately from the nine
    # real topics (see TOPIC_CLASSES) and never averaged into a macro-F1 as
    # though it were one -- that is what inflated the 0.856 headline.
    "international",  # placement: the outlet's foreign desk, not a subject
]

# Labels carrying no topical signal. Kept explicit rather than implicit so the
# decision is reviewable.
_DROP_RAW = {
    # generic "news"
    "မြန်မာသတင်း", "သတင်း", "သတင်းများ", "မြန်မာ", "News",
    # format / placement / section furniture
    "Lead Story", "Video", "ရုပ်သံ", "ရုပ်သံသတင်း", "ဆောင်းပါး", "Feature",
    "Cartoon", "သရော်စာ", "Data Story", "Weekend Reading", "သတင်းတို",
    "မာလ်တီမီဒီယာ", "ဗီဒီယိုများ", "သတင်းဓာတ်ပုံ", "ရေဒီယို", "photos",
    "Photo", "အင်တာဗျူး", "မေးမြန်းခန်း", "Interview", "RFA စကားဝိုင်း",
    "RFA ပေးစာ ပြန်စာအပတ်စဉ်ကဏ္ဍများ", "special-section", "Obituary",
    # opinion: a stance marker, not a topic. Excluded from the topic target;
    # the raw label stays in the data if you later want an opinion/news task.
    "သူ့အတွေး သူ့အမြင်", "သုံးသပ်ချက်", "အတွေးအမြင်", "Opinion", "Editorial",
    "Viewpoint", "Analysis", "Commentary", "Guest Column", "Letter",
    # domestic place names: location, not subject
    "ရန်ကုန်", "မန္တလေး", "နေပြည်တော်",
}

MAPPING: dict[str, str] = {}


def _norm(s: str) -> str:
    """NFC-normalise a label before matching.

    Burmese combining marks can legitimately appear in either order and render
    identically: the corpus writes U+103A (asat) before U+1037 (dot below),
    while the same label typed elsewhere may carry them reversed. Those are
    different byte strings, so a plain dict lookup silently misses -- which it
    did here, dropping ~1,000 label assignments before this was caught.
    Normalising both sides is the fix.
    """
    import unicodedata
    return unicodedata.normalize("NFC", s)


def _add(cls: str, *labels: str) -> None:
    for l in labels:
        MAPPING[_norm(l)] = cls


DROP = {_norm(l) for l in _DROP_RAW}


_add("conflict",
     "ငြိမ်းချမ်းရေး/စစ်ရေး", "စစ်တပ်", "မြန်မာပြည်တွင်းစစ်", "Military",
     "ပြည်သူ့ကာကွယ်ရေးတပ်မတော်", "အာရက္ခတပ်တော်", "ကချင်လွတ်လပ်ရေးအဖွဲ့",
     "ကရင်အမျိုးသား အစည်းအရုံး", "တအာင်းအမျိုးသားလွတ်မြောက်ရေးတပ်မတော်",
     "ဗိုလ်ချုပ်မှူးကြီး မင်းအောင်လှိုင်", "peace-war", "Conflict",
     "စစ်ဘေးရှောင်", "လက်နက်ကိုင်", "အာဏာသိမ်းမှု")

_add("politics",
     "နိုင်ငံရေး", "Politics", "ရွေးကောက်ပွဲ", "Election",
     "အထွေထွေ ရွေးကောက်ပွဲ၊ ၂၀၂၅-၂၀၂၆", "၂၀၂၀ ရွေးကောက်ပွဲ",
     "အမျိုးသားညီညွတ်ရေးအစိုးရ", "ဒေါ်အောင်ဆန်းစုကြည်",
     "လွှတ်တော်/ဥပဒေရေးရာ", "ဥပဒေ", "politics-new", "2012election",
     "my-vote", "အစိုးရ", "Government", "Law")

_add("crime",
     "မှုခင်း/မူးယစ်ဆေး", "မှုခင်း", "Crime", "criminals-news", "drugs",
     "ကျားဖြန့် (အွန်လိုင်းငွေလိမ်လုပ်ငန်း)", "မူးယစ်ဆေးဝါး",
     "လူကုန်ကူးမှု", "Drugs")

_add("economy",
     "စီးပွားရေး", "စီးပွားရေးလုပ်ငန်း", "ကုန်သွယ်ရေး",
     "စိုက်ပျိုးရေး/တောင်သူလယ်သမား", "အလုပ်သမား/ရွှေ့ပြောင်းလုပ်သား",
     "ရွှေ့ပြောင်း အလုပ်သမား", "Labor Rights", "labour-news",
     "ရေနံ နှင့် သဘာဝဓာတ်ငွေ့ လုပ်ငန်း", "Business", "Economy",
     "farmers-news", "economics-new", "ကျပ်ငွေ", "ရင်းနှီးမြှုပ်နှံမှု")

_add("environment",
     "သဘာဝပတ်ဝန်းကျင်/ရာသီဥတု/မိုးလေဝသ", "Environment", "ငလျင်",
     "၂၀၂၅ မြန်မာငလျင်", "ပြင်းထန်သည့် ရာသီဥတု", "environment-weather",
     "ရာသီဥတု", "သဘာဝဘေး", "Climate", "Disaster")

_add("health",
     "ကျန်းမာရေး", "health-news-news", "covid19", "Health", "ကိုဗစ်")

_add("education",
     "ပညာရေး/နည်းပညာ", "ပညာရေး", "နည်းပညာ", "education-news",
     "Education", "Technology", "academy")

# Split out of a single over-broad bucket: the first version lumped rights,
# refugees, women/youth and general social affairs together and scored F1 0.274
# because they do not cohere. Ethnic affairs sits with rights rather than
# society: in Myanmar reporting it is a rights-and-autonomy frame.
# Rights, ethnic affairs and social coverage all fold into politics -- see the
# note in CLASSES for the measurements behind that decision.
_add("politics",
     "လူ့အခွင့်အရေး", "Human Rights", "humanright",
     "ဒုက္ခသည်များ နှင့် ခိုလှုံခွင့်တောင်းသူများ",
     "လူသားချင်းစာနာမှုဆိုင်ရာ အကူအညီ", "တိုင်းရင်းသား",
     "လူမျိုး နှင့် တိုင်းရင်းသားရေးရာ", "Ethnic Issues", "ethnic",
     "ရိုဟင်ဂျာ", "Refugees",
     "အမျိုးသမီး/လူငယ်", "အမျိုးသမီး", "လူငယ်", "women-youth",
     "လူမှုရေး", "social-news", "Society", "ကလေးသူငယ်")

_add("sport",
     "အားကစား", "ဘောလုံး", "၂၀၂၆ ကမ္ဘာ့ဖလား", "sports-news", "football",
     "tennis", "one-championship", "2026-worldcup", "Sport")

_add("culture",
     "အနုပညာ", "Culture", "ဘာသာရေး", "religion", "art", "movie-crazy",
     "book-review", "thingyanfastival", "လူနေမှုဘဝ", "Lifestyle",
     "ဘုရားဖူး", "ရိုးရာပွဲ")

_add("international",
     "နိုင်ငံတကာ", "international-news", "အမေရိကန်", "အမေရိကန်အစိုးရ",
     "တရုတ်", "ထိုင်း", "အီရန်", "ရုရှား", "အစ္စရေး", "ယူကရိန်း",
     "အိန္ဒိယ", "ဗြိတိန်", "ဂျပန်", "ဥရောပ", "အာရှ", "အာဆီယံ",
     "အရှေ့အလယ်ပိုင်း", "အစ္စရေး နှင့် ပါလက်စတိုင်း", "ဒေါ်နယ်ထရမ့်",
     "ဗလာဒီမီယာ ပူတင်", "asia-region", "china-focus", "International",
     "ဘင်္ဂလားဒေ့ရှ်", "မလေးရှား", "စင်္ကာပူ", "ကိုရီးယား")

# The nine actual topics. `international` is excluded because it is a placement
# label, so averaging it into a topic metric reports something that is not a
# topic score. See the note in CLASSES for the measurement behind this.
TOPIC_CLASSES = [c for c in CLASSES if c != "international"]

# Media coverage of media is its own thing; treat as culture rather than
# inventing a class for ~100 articles.
_add("culture", "သတင်းမီဒီယာ", "media-news", "Media")

# Most specific first. An article tagged both conflict and politics is
# conflict; one tagged both economy and international is economy.
#
# `international` is last, and that is the whole point: it only wins when
# nothing else applies. A foreign story with an identifiable subject takes the
# subject -- ['နိုင်ငံတကာ', 'ငလျင'] is environment, not international.
RESOLUTION_ORDER = [
    "conflict", "crime", "environment", "health", "sport",
    "economy", "education", "politics", "culture", "international",
]

_PRIORITY = {c: i for i, c in enumerate(RESOLUTION_ORDER)}


def label_article(labels) -> str | None:
    """Collapse an article's outlet labels to one topic class, or None.

    None means "no usable topical signal" -- the article is excluded from the
    dataset rather than forced into a class.
    """
    norm = [_norm(l) for l in (labels or [])]
    hits = {MAPPING[l] for l in norm if l in MAPPING and l not in DROP}
    if not hits:
        return None
    return min(hits, key=lambda c: _PRIORITY[c])


def coverage_report(all_labels_with_counts):
    """Which labels are mapped, dropped, or unaccounted for."""
    mapped = unmapped = dropped = 0
    missing = []
    for raw, n in all_labels_with_counts:
        label = _norm(raw)
        if label in DROP:
            dropped += n
        elif label in MAPPING:
            mapped += n
        else:
            unmapped += n
            missing.append((raw, n))
    return {
        "mapped": mapped,
        "dropped": dropped,
        "unmapped": unmapped,
        "top_unmapped": sorted(missing, key=lambda t: -t[1])[:25],
    }

def fingerprint() -> str:
    """Stable hash of the mapping that defines a dataset's labels.

    build_dataset stamps this into its output; anything consuming that output
    checks it. Without this the failure is silent: taxonomy.py is read at
    runtime but train.jsonl is a materialised artefact on disk, so editing the
    taxonomy leaves a stale dataset behind that still loads, still trains and
    still reports plausible-looking metrics for a taxonomy no longer in force.
    That happened here -- a prediction run got most of the way through on an
    11-class dataset after rights/society had been folded into politics, and
    the only tell was a macro-F1 that matched the superseded number.
    """
    import hashlib
    payload = "\u0000".join([
        "|".join(CLASSES),
        "|".join(RESOLUTION_ORDER),
        "|".join(f"{k}={v}" for k, v in sorted(MAPPING.items())),
        "|".join(sorted(DROP)),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
