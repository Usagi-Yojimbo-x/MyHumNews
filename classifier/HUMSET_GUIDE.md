# HumSet labelling guide — 383 articles, multi-label

    python3 -m classifier.label_web --multi \
        --task classifier/data/humset_task.jsonl \
        --out  classifier/data/humset_labels.jsonl

**Pass `--multi` and both paths.** Without them you reopen an old single-label set.

## This works differently from the last two passes

Digits now **toggle**. Nothing is saved until you press **↵**.

    1  Context                   5  Humanitarian Access
    2  Shock / Event             6  Information & Comms
    3  Displacement              0  Other  (exclusive)
    4  Casualties

    ↵  commit the selected set and move on
    a  ambiguous      s  not an article      b  undo      ?  rules

Selected labels turn green and are listed under the progress bar. `other`
clears everything else, because it means *no humanitarian content at all*.

## The labels

| label | what it covers |
|---|---|
| **context** | the surrounding situation, not an event: politics, economy, demography, law, security, socio-cultural background. Analysis and explainers. |
| **shock_event** | the hazard or incident itself — an airstrike, earthquake, flood, raid, battle. *What happened.* |
| **displacement** | people moving or unable to: fleeing, IDPs, refugees, camps, returns, numbers displaced. |
| **casualties** | dead, injured, missing. Counts or named victims. |
| **humanitarian_access** | aid reaching people or failing to: blocked roads, denied permission, sieges, aid suspension, relief operations, who is cut off. |
| **information_communication** | information *as the subject*: internet or phone shutdowns, censorship, press freedom, journalists detained, rumour. |
| **other** | no humanitarian relevance. Sport, entertainment, celebrity, routine business. **Exclusive.** |

## Multiple labels are the normal case

> `လေးမျက်နှာမြို့နယ်ကို စစ်တပ်က ဗုံးကြဲ၊ အရပ်သား ၁ ဦး သေဆုံး`
> Military bombs Lemyethna township, 1 civilian killed
> → **shock_event** + **casualties**

> `ထိုင်း-မြန်မာနယ်စပ်အနီး တိုက်ပွဲကြောင့် ဒေသခံ လေးရာကျော် ထိုင်းဘက်ထွက်ပြေး`
> Fighting near the Thai border, 400+ locals flee into Thailand
> → **shock_event** + **displacement**

> `မကွေးတိုင်းမှာ စစ်ဘေးဒုက္ခသည် တစ်သိန်းကျော် အရေးပေါ်အကူအညီလိုအပ်နေ`
> 100,000+ war-displaced in Magway need emergency aid
> → **displacement** + **humanitarian_access**

Do not pick one and move on. Under-labelling costs recall on every label you
omit, and per-label recall is the headline metric in the proposal.

## The four confusions that matter

**shock_event vs casualties** — they co-occur constantly and that is correct.
"Airstrike kills 3" is both: the airstrike is the event, the deaths are
casualties. Never choose between them.

**context vs everything else** — `context` is the backdrop, not the incident. A
battle is `shock_event`. An analysis of *why* the offensive is happening is
`context`. If the article reports something that happened, it is probably not
`context` alone. Resist using it as a catch-all; it is the label most likely to
end up meaningless.

**displacement vs humanitarian_access** — displacement is *people* moving;
access is *aid* moving, or not. "20,000 flee to the border" is displacement.
"Aid convoys blocked from reaching them" is access. Many articles are both.

**information_communication** — only when information *is* the subject. An
internet shutdown qualifies. An article that merely quotes a Facebook post does
not.

## READ THIS FIRST — `other` vs `context`

**`other` means no humanitarian relevance *at all*.** Not "no emergency", not
"nothing dramatic happened". It is for articles a person tracking Myanmar's
humanitarian situation would have no reason to read.

Myanmar politics, economy, sanctions, diplomacy, detention, elections and trade
are **`context`**, never `other`. HumSet's context pillar covers exactly this:
politics, economy, demography, law and policy, security and stability.

**The test:** *would someone monitoring Myanmar's humanitarian situation want to
know this?* If yes, it is not `other`.

| article | label | not |
|---|---|---|
| Junta leader thanks China for support | `context` | ~~other~~ |
| Switzerland sanctions MOGE | `context` | ~~other~~ |
| Myanmar economy forecast to contract 2% | `context` | ~~other~~ |
| ASSK's son demands her release | `context` | ~~other~~ |
| Youth disengaged from the junta's election | `context` | ~~other~~ |
| Military blocks Chinese goods in northern Shan | `context` | ~~other~~ |
| Yangon air pollution index rising | `context` | ~~other~~ |
| Thailand won't ease fuel/power restrictions | `context` + `humanitarian_access` | ~~other~~ |
| Myawaddy cut off from Thai fuel; panic buying | `context` + `humanitarian_access` | ~~other~~ |
| 8 Myanmar cities among the world's hottest | `shock_event` + `context` | ~~other~~ |

**`other` genuinely is:** Premier League results · World Cup predictions · a
Spider-Man box office record · an octopus picking match winners · a Norwegian
royal funeral · a meteor over New England · Coupang fined for a data breach ·
foreign naval exercises with no Myanmar angle.

The dividing line is **Myanmar relevance**, not drama. A dull story about
Myanmar's electricity tariffs is `context`. An exciting story about an Italian
stadium is `other`.

## Use `other` freely

Roughly a third of DVB's output is sport, entertainment and foreign general
news with no humanitarian angle — your sample has ~28 sport articles alone. The
proposal needs `other` to keep those out of the RAG index, so it has a real
function.

But note which way the error costs more: **a false `other` silently removes an
article from the whole system.** Nothing downstream ever surfaces it again. A
wrong topic can be corrected later; a wrongly excluded article cannot, because
nobody looks at it. When genuinely torn between `other` and `context`, choose
`context`.

## `a` and `s` still mean different things

- **`s`** — not an article. Programme announcements, boilerplate.
  *(17 already removed from this task; you should hit few or none.)*
- **`a`** — a real article the label set cannot express.

Guessing on an `a` case corrupts the measurement; `a` is reported as a bound.
If `a` runs above ~10%, the label set needs work before the numbers mean much.

## Why these 383

They are the 400 you already labelled (200 RFA + 200 DVB held-out), minus 17
confirmed non-articles. Two advantages: you have read them, so this pass is
faster; and the DVB half is held-out test data the model never trained on, so
it can serve directly as the hand-checked test set slide 5 requires.

The previous single-label judgements are **not** shown — they were against a
different taxonomy and would only anchor you.

---

Earlier task-specific rounds are archived in `HUMSET_GUIDE_round2_archive.md`.
Label what is actually in the article. There is no quota and no label this pass
is trying to collect.
