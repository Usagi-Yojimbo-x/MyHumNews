#!/usr/bin/env python3
"""Other outlets' categories mapped onto the DVB scheme, plus one new label.

    .venv/bin/python dataset/outlet_crosswalk.py --verify Military
    .venv/bin/python dataset/outlet_crosswalk.py            # coverage report

dvb_scheme.py matches DVB's Burmese category strings. The other four outlets
file against their own vocabularies -- Irrawaddy in English, BBC in a much
finer-grained Burmese taxonomy -- so their topical categories were invisible
to it. Irrawaddy's `Military` is the same desk as DVB's ငြိမ်းချမ်းရေး/စစ်ရေး;
it simply does not string-match.

Every label this recovers is PUBLISHER-ASSIGNED. It replaces a model
prediction (cross-outlet precision 0.645-0.979, mostly untestable) with an
editor's own filing decision, which is strictly better and is why this runs
before the classifier rather than after it.

## One new label

`လူ့အခွင့်အရေး` (human_rights) is added as a 20th label. DVB has no human
rights desk, but Irrawaddy and BBC both do, and in a humanitarian corpus the
topic is too valuable to collapse into politics or crime. It is the only label
in the scheme with no DVB support, which has two consequences worth stating:
the temporal-split baselines do not cover it, and the classifier cannot
predict it onto unlabelled articles.

## What is deliberately NOT mapped, and why

FORMAT AND GENERIC -- `Video`, `Feature`, `Cartoon`, `သတင်း` (news),
`ဆောင်းပါး` (article), `မေးမြန်းခန်း` (interview), `သတင်းများ`,
`မာလ်တီမီဒီယာ`. Same reasoning as dvb_scheme's DROP_FORMAT: they record how a
story was presented, not what it is about. This is why RFA recovers nothing --
21,251 of its 23,534 articles are filed under `သတင်းများ` ("news").

PERSON AND ORGANISATION TAGS -- `ဗိုလ်ချုပ်မှူးကြီး မင်းအောင်လှိုင်` (880),
`ဒေါ်အောင်ဆန်းစုကြည်`, `ဒေါ်နယ်ထရမ့်`, `ဗလာဒီမီယာ ပူတင်`, `ကုလသမဂ္ဂ`,
`အာဆီယံ`, `အမျိုးသားညီညွတ်ရေးအစိုးရ`. BBC tags the entities a story mentions,
which is a different axis from the desk it belongs to. An article tagged
`မင်းအောင်လှိုင်` might be military, politics or crime.

`မြန်မာ` (4,902) -- BBC's DOMESTIC marker, the complement of the geographic
tags below. It carries no topic: it only says the story is about Myanmar.

NO DVB EQUIVALENT -- `လူသားချင်းစာနာမှုဆိုင်ရာ အကူအညီ` (humanitarian
assistance, 271) and `ဒုက္ခသည်များ နှင့် ခိုလှုံခွင့်တောင်းသူများ` (refugees
and asylum seekers, 479). Both are squarely humanitarian, and neither maps:
DVB's အလုပ်သမား/ရွှေ့ပြောင်းလုပ်သား is migrant LABOUR, not displacement.
Inventing labels for them would be adding my own taxonomy to a scheme whose
whole point is that it is a publisher's. They are left unmapped and recorded
here so the omission is visible rather than silent.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import dvb_scheme as S  # noqa: E402

HUMAN_RIGHTS = "လူ့အခွင့်အရေး"
PEACE = "ငြိမ်းချမ်းရေး/စစ်ရေး"
INTL = "နိုင်ငံတကာ"
POLITICS = "နိုင်ငံရေး"
ETHNIC = "တိုင်းရင်းသား"
ENV = "သဘာဝပတ်ဝန်းကျင်/ရာသီဥတု/မိုးလေဝသ"
HEALTH = "ကျန်းမာရေး"
EDU = "ပညာရေး/နည်းပညာ"
ART = "အနုပညာ"
ECON = "စီးပွားရေး"
LABOUR = "အလုပ်သမား/ရွှေ့ပြောင်းလုပ်သား"
ELECTION = "ရွေးကောက်ပွဲ"
WOMEN = "အမျိုးသမီး/လူငယ်"
SOCIAL = "လူမှုရေး"

# Each entry: category -> (label, why). The reason is the point: a mapping
# without one is a guess, and this file replaces guesses with editors.
CROSSWALK: dict[str, tuple[str, str]] = {
    # ---- Irrawaddy, English desks that are plainly DVB desks -------------
    "Politics":      (POLITICS, "same desk, English name"),
    "Military":      (PEACE, "DVB files war under peace/military; same desk"),
    "Ethnic Issues": (ETHNIC, "same desk, English name"),
    "Crime":         ("crime", "same desk, English name"),
    "Environment":   (ENV, "same desk, English name"),
    "Health":        (HEALTH, "same desk, English name"),
    "Entertainment": (ART, "DVB's အနုပညာ is the arts/entertainment desk"),
    "Labor Rights":  (LABOUR, "labour desk; the rights framing is Irrawaddy's"),
    "Human Rights":  (HUMAN_RIGHTS, "the new label this file adds"),
    "Coronavirus Update": ("COVID19", "same subject, outlet's own wording"),

    # ---- BBC, Burmese but a different vocabulary -------------------------
    "စစ်တပ်":                           (PEACE, "military"),
    "မြန်မာပြည်တွင်းစစ်":                (PEACE, "Myanmar civil war"),
    "လူမျိုး နှင့် တိုင်းရင်းသားရေးရာ":  (ETHNIC, "ethnic and nationality affairs"),
    "လူ့အခွင့်အရေး":                     (HUMAN_RIGHTS, "human rights"),
    "အမျိုးသမီး အခွင့်အရေး":             (HUMAN_RIGHTS, "women's rights"),
    "လွှတ်တော်":                         (POLITICS, "parliament"),
    "၂၀၂၀ ရွေးကောက်ပွဲ":                 (ELECTION, "the 2020 election"),
    "အထွေထွေ ရွေးကောက်ပွဲ၊ ၂၀၂၅-၂၀၂၆":   (ELECTION, "the 2025-26 general election"),
    "ရာဇဝတ်မှု":                         ("crime", "crime"),
    "ရာသီဥတုပြောင်းလဲမှု":               (ENV, "climate change"),
    "ပြင်းထန်သည့် ရာသီဥတု":              (ENV, "extreme weather"),
    "ငလျင်":                             (ENV, "earthquake; DVB files disasters here"),
    "ကိုရိုနာဗိုင်းရပ်စ်":               ("COVID19", "coronavirus"),
    "ပညာရေး":                            (EDU, "education"),
    "နည်းပညာ":                           (EDU, "DVB's desk is education/technology"),
    "စီးပွားရေးလုပ်ငန်း":                (ECON, "business"),
    "ကုန်သွယ်ရေး":                       (ECON, "trade"),
    "အမျိုးသမီး":                        (WOMEN, "women"),
    "ကလေးသူငယ်":                         (WOMEN, "DVB's desk is women/youth"),
    "လူမှုဘဝ":                           (SOCIAL, "social life"),
    "ရဲလုပ်ငန်း":                        ("crime", "policing"),

    # BBC geographic tags -> the foreign desk. This is exactly what DVB's
    # နိုင်ငံတကာ means: not-Myanmar. Recorded in dvb_scheme as a placement
    # label rather than a subject, and that caveat carries over here.
    "တရုတ်":              (INTL, "China"),
    "အမေရိကန်":           (INTL, "United States"),
    "ရုရှား":             (INTL, "Russia"),
    "ယူကရိန်း":           (INTL, "Ukraine"),
    "ထိုင်း":             (INTL, "Thailand"),
    "အိန္ဒိယ":            (INTL, "India"),
    "ဗြိတိန်":            (INTL, "Britain"),
    "အာရှ":               (INTL, "Asia"),
    "အစ္စရေး":            (INTL, "Israel"),
    "အစ္စရေး နှင့် ပါလက်စတိုင်း": (INTL, "Israel and Palestine"),
    "ဥရောပ":              (INTL, "Europe"),
    "အရှေ့အလယ်ပိုင်း":    (INTL, "Middle East"),
    "အီရန်":              (INTL, "Iran"),
    "ဂျပန်":              (INTL, "Japan"),
    "တောင်ကိုရီးယား":     (INTL, "South Korea"),
    "အရှေ့တောင်အာရှ":     (INTL, "Southeast Asia"),
}

# Recorded so the omissions are auditable rather than invisible.
NOT_MAPPED = {
    "Culture": "VERIFIED UNMAPPABLE -- sampled 4, all also tagged ဆောင်းပါး "
               "(feature), which dvb_scheme drops as format. Contents were a "
               "Buddhist essay, a correspondent's memoir, a religious "
               "festival and university nostalgia. It is Irrawaddy's essay "
               "section, not an arts desk. I had mapped it to အနုပညာ and "
               "verification said no.",
    "Lifestyle": "VERIFIED UNMAPPABLE -- co-occurs with Entertainment, "
                 "Culture AND Coronavirus Update; samples ranged from an "
                 "artist's detention to Putin's vaccination. A soft-features "
                 "bucket spanning unrelated subjects. I had mapped it to "
                 "လူမှုရေး, which would have injected noise into a label "
                 "already too inconsistent to propagate.",
    "မြန်မာ": "BBC's domestic marker; the complement of the geographic tags",
    "လူသားချင်းစာနာမှုဆိုင်ရာ အကူအညီ": "humanitarian assistance -- no DVB desk",
    "ဒုက္ခသည်များ နှင့် ခိုလှုံခွင့်တောင်းသူများ":
        "refugees/asylum -- DVB's labour desk is migrant WORK, not displacement",
    "ရိုဟင်ဂျာ": "Rohingya -- an affected group, not a desk",
    "ဗိုလ်ချုပ်မှူးကြီး မင်းအောင်လှိုင်": "person tag",
    "ဒေါ်အောင်ဆန်းစုကြည်": "person tag",
    "ဒေါ်နယ်ထရမ့်": "person tag",
    "ဗလာဒီမီယာ ပူတင်": "person tag",
    "ကုလသမဂ္ဂ": "organisation tag",
    "အာဆီယံ": "organisation tag",
    "အမျိုးသားညီညွတ်ရေးအစိုးရ": "organisation tag",
    "အမျိုးသားဒီမိုကရေစီ အဖွဲ့ချုပ်": "organisation tag",
    "ကန့်ကွက်ဆန္ဒပြခြင်း": "protest -- spans politics and social; no clean desk",
    "ဆန္ဒထုတ်ဖော်ခြင်း": "protest -- as above",
    "ဥပဒေ": "law -- spans politics and crime",
    "သမိုင်း": "history -- not a news desk",
    "ရန်ကုန်": "place tag",
}


def labels(categories) -> list[str]:
    """DVB scheme labels plus anything the crosswalk recovers."""
    out = {l for l in S.labels(categories) if l in S.F1_TEMPORAL}
    for c in categories or []:
        hit = CROSSWALK.get((c or "").strip())
        if hit:
            out.add(hit[0])
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path,
                    default=Path("dataset/myhumnews.jsonl"))
    ap.add_argument("--verify", help="print sampled articles for one category")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", type=Path,
                    default=Path("dataset/dvb_crosswalk_report.json"))
    args = ap.parse_args()

    if args.verify:
        rng = random.Random(20261024)
        pool = []
        for line in args.release.open():
            d = json.loads(line)
            if args.verify in (d.get("categories") or []):
                pool.append(d)
        lab, why = CROSSWALK.get(args.verify, ("(unmapped)", ""))
        print("category %r -> %s   (%s)" % (args.verify, lab, why))
        print("%d articles carry it; sampling %d\n" % (len(pool), args.n))
        for d in rng.sample(pool, min(args.n, len(pool))):
            print("  %s  %s" % (d["published_date"][:10], d["outlet"]))
            print("    %s" % (d.get("title") or "")[:110])
            print("    %s" % (d.get("body") or "")[:170].replace("\n", " "))
            print("    categories: %s" % ", ".join(d.get("categories") or []))
            print()
        return 0

    before = after = 0
    gained = collections.Counter()
    by_outlet = collections.Counter()
    new_label = collections.Counter()
    n = 0
    for line in args.release.open():
        d = json.loads(line); n += 1
        cats = d.get("categories") or []
        old = set(d.get("topic_dvb19") or [])
        new = set(labels(cats))
        if old:
            before += 1
        if new:
            after += 1
        if new and not old:
            by_outlet[d["outlet"]] += 1
        for l in new - old:
            gained[l] += 1
            if l == HUMAN_RIGHTS:
                new_label[d["outlet"]] += 1

    print("publisher-labelled before crosswalk  %s (%.1f%%)"
          % (format(before, ","), 100 * before / n))
    print("publisher-labelled after crosswalk   %s (%.1f%%)"
          % (format(after, ","), 100 * after / n))
    print("newly labelled articles              %s" % format(after - before, ","))
    print()
    print("articles gaining a first label, by outlet:")
    for o, c in by_outlet.most_common():
        print("   %-22s %s" % (o, format(c, ",")))
    print()
    print("label assignments gained:")
    for l, c in gained.most_common():
        print("   %-36s %s" % (l, format(c, ",")))
    print()
    print("the new label, %s:" % HUMAN_RIGHTS)
    for o, c in new_label.most_common():
        print("   %-22s %s" % (o, format(c, ",")))

    json.dump({"n_articles": n, "before": before, "after": after,
               "gained_by_label": dict(gained),
               "first_label_by_outlet": dict(by_outlet),
               "new_label": HUMAN_RIGHTS,
               "new_label_by_outlet": dict(new_label),
               "crosswalk": {k: list(v) for k, v in CROSSWALK.items()},
               "not_mapped": NOT_MAPPED}, args.out.open("w"),
              ensure_ascii=False, indent=1)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
