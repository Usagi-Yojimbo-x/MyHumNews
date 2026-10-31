# Labelling guide

Keep this open while labelling. Every rule below was derived from what the
**gold labels actually do** (29,206 articles), not from what seems sensible —
your labels have to match the taxonomy as it was applied to the training data,
or `score_labels` measures your disagreement with the taxonomy instead of the
model's error.

## READ THIS FIRST — `international` almost never applies

**`international` means foreign news with NO Myanmar angle.** It is **last** in
`RESOLUTION_ORDER`, so it only wins when nothing else fits.

Measured in the gold data: of 8,674 `international` articles, only **1.6%** even
mention Myanmar in the title. Articles with both Myanmar and a foreign actor go
elsewhere — politics 28%, economy 21%, conflict 19%, crime 15%, and
`international` just 4.9%.

So a foreign country doing something **about Myanmar** is *not* international:

| title | NOT international — it is |
|---|---|
| Vietnamese PM hopes Myanmar stabilises | **politics** (diplomacy over Myanmar) |
| Switzerland sanctions MOGE | **economy** (sanctions on an energy firm) |
| US travel ban includes Myanmar | **politics** |
| Britain urges action on military atrocities | **politics** |
| ASEAN envoy to be allowed to meet ASSK | **politics** |
| China closes border trade in northern Shan | **economy** |
| UN says a third of Myanmar needs aid | **politics** (humanitarian) |

**`international` is for:** Russia testing nuclear weapons · Chile's desert fog
farms · India's police shooting student protesters · a lunar eclipse · Trump
banning Chinese robots. Stories where Myanmar simply does not appear.

Ask: **if you deleted every mention of Myanmar, would the story survive?**
If yes → international. If it collapses → it is a Myanmar story, labelled by
its subject.

## The one rule that resolves most hard cases

**Label the consequence being reported, not the cause.**

Myanmar's war is the backdrop to almost everything, so "is the war involved?"
is not a usable question — the answer is nearly always yes. Ask instead: *what
does this article actually report on?*

## Keys

    1 conflict   2 politics   3 crime      4 economy     5 environment
    6 health     7 education  8 sport      9 culture     0 international

    a  ambiguous — a real article, but the taxonomy genuinely doesn't decide it
    s  skip      — not an article (programme announcement, boilerplate)
    b  undo

Use `a` rather than guessing. Guesses are indistinguishable from model errors
and silently corrupt the measurement; `a` is counted and reported as a bound.

## conflict — what it covers

Armed violence, the military or an armed group as actor, the war itself.
Mapped labels: ငြိမ်းချမ်းရေး/စစ်ရေး · စစ်တပ် · မြန်မာပြည်တွင်းစစ် · PDF · AA ·
KIA · KNU · TNLA · စစ်ဘေးရှောင် · လက်နက်ကိုင် · အာဏာသိမ်းမှု

It is **first in RESOLUTION_ORDER**, so where conflict and politics both apply,
conflict wins. An article about fighting that is also about politics is conflict.

## Conscription (စစ်မှုထမ်း) — n=441 in gold

The gold labels are consistent here, and the determinant is the consequence:

| the article reports | label |
|---|---|
| military forcibly rounding people up, conscripting by force | **conflict** |
| the conscription law, arrests under it, fees/quotas levied | **politics** |
| workers quitting jobs, employers pressuring staff out | **economy** |
| students leaving for Thailand, schooling disrupted | **education** |
| another country's military service or exercises | **international** |
| a bomb at a conscription office | **crime** |

Worked example: `စစ်မှုထမ်းရမည်ကို စိုးရိမ်ပြီး လှိုင်သာယာစက်မှုဇုန်က အလုပ်သမားများ
အလုပ်ထွက်` → **economy**. Conscription is the cause; labour is the subject.

## Landmines (မိုင်း / မြေမြှုပ်မိုင်း) — n=62 in gold

- Casualties, mine risk, clearance → **conflict** (56% of gold)
- Framed around displaced/refugee populations → **politics** (31%)
- A mine abroad → **international**

Landmines are weapons of the war, so conflict is the default.

## Mining deaths — USE `a`

Distinguish two different things:

**Military attacking a mining area** → **conflict**, unambiguously. The military
is the actor. This is most Hpakant coverage (45% of such titles).

**An industrial accident** — a jade or gold mine collapsing, a mud pond bursting,
scavengers killed — **has no consistent gold label**. The same event type appears
as three different classes:

    [crime      ] ဖားကန့်တွင် နွံကန်ပြိုကျမှုကြောင့် သေဆုံးသူ ၈ ဦးအထိရှိလာ
    [politics   ] ဖားကန့်မှာ နွံကန်ပြိုကျပြီး လူ ၁၀ ဦးထက်မနည်း သေဆုံး
    [environment] ဖားကန့်တွင် နွံကန် ပေါက်ပြီး ကားလမ်းပိတ်၊ သွားလာရေးခက်ခဲ

Gold distribution for jade/ore-mine titles: conflict 45%, politics 15%,
crime 12%, environment 11%, international 11%, economy 5%.

No label can be "correct" against that. Press `a`.

The underlying cause is a real gap: `environment` is defined as "weather,
climate, earthquakes, disasters" and an industrial accident is not a natural
disaster, so the taxonomy has nowhere to put it. Worth deciding deliberately
after this measurement — either add an `accident`/`industry` class or write an
explicit rule — but do not paper over it now by guessing.

## A known taxonomy inconsistency

`taxonomy.py` maps စစ်ဘေးရှောင် (war-displaced) → **conflict** but
ဒုက္ခသည်များ (refugees) → **politics**. Both mean displaced people, so otherwise
identical articles get different gold labels depending on the outlet's word
choice.

While labelling, go by the **substance**: displacement caused by fighting is
conflict. If the article is about refugee policy, asylum or resettlement, that
is politics. Flag with `a` if it is genuinely both.

## Other calls worth fixing in advance

| case | label | why |
|---|---|---|
| Earthquake damage, aftermath, relief | environment | ငလျင် maps there |
| Scam compounds, trafficking, drugs | crime | even though armed groups run them |
| Rohingya, ethnic affairs, human rights | politics | folded in deliberately |
| Aid, humanitarian assistance | politics | လူသားချင်းစာနာမှု maps there |
| Myanmar migrant workers abroad | economy | labour is the subject |
| Myanmar at an ASEAN/UN meeting | politics | the diplomacy is about Myanmar |
| Foreign govt sanctioning / pressuring Myanmar | politics | see the section at the top |
| Foreign sanctions on a Myanmar company | economy | the firm is the subject |
| Foreign news with no Myanmar angle | international | the only case for this class |
| Religion, festivals, arts, obituaries of artists | culture | |

## `s` and `a` are not interchangeable

- **`s` skip** — *not an article*. Programme announcements
  (`RFA ရေဒီယိုနဲ့ရုပ်သံ အထူးအစီအစဉ်...`), boilerplate, stubs under ~300 chars.
- **`a` ambiguous** — *a real article* the taxonomy cannot decide.

This matters more than it looks. Skips are dropped from scoring entirely, so
using `s` on a real article you found hard removes it from the measurement and
leaves only the easy ones — which flatters the model instead of measuring it.
`a` keeps the article in the count and is reported as an explicit bound.

If it has a byline, a dateline and paragraphs of prose, it is an article. Label
it, or press `a`. Never `s`.

## When you are still stuck

Press `a`. Two or three per hundred is normal and useful. Thirty would be a
signal that the taxonomy needs work before any of these numbers mean anything —
which is itself worth knowing, and is exactly what `score_labels` will tell you.

---

# DVB sample (200, held-out)

    python -m classifier.label_web \
        --task classifier/data/dvb_task.jsonl \
        --out  classifier/data/dvb_labels.jsonl

**Pass both flags.** Without them it reopens the finished RFA set.

## What this sample is for

Unlike RFA, DVB *has* gold labels — and they are DVB's own section names. So
`dvb 0.875` means "the model reproduces DVB's desk conventions 87.5% of the
time", which is not the same claim as "it identifies the topic correctly".

These 200 are drawn from the **held-out test set** (5,060 articles the model
never trained on, 2026-05-25 to 2026-09-23), which lets all three be measured:

    model vs hand   the model's real accuracy
    gold  vs hand   how good DVB's own section names are  <- the point
    model vs gold   0.875, the number usually quoted

If gold agrees with you ~90% of the time, the headline figure is sound. If it
agrees ~70%, then 0.875 is largely the model reproducing a noisy label source,
and most of this project's numbers need restating. **Either answer is worth
having, so label what you see and do not try to guess what DVB would have
said.**

## Label the article, not the section

This is the one way this pass differs from RFA. The temptation is to think
"which DVB section would this have run in?" — that defeats the entire purpose,
because agreement with DVB's sections is precisely what is being tested.

Judge the article on its content, by the taxonomy. If your label disagrees with
DVB's, that disagreement **is the measurement**.

## Expect a lot of international

DVB's held-out gold mix runs `international` 34.6%, conflict 15.1%, sport 14.4%,
politics 12.1%. A random 200 should land near 69 international and 29 sport —
very different from RFA.

The `international` rule at the top of this guide matters more here than
anywhere: **foreign news with no Myanmar angle.** DVB runs a large foreign desk,
so most of those 69 should be genuine world news (a Thai election, a US tariff,
a European heatwave). A foreign government acting *about* Myanmar is still
politics.

## Sport

RFA had essentially none; DVB has ~29 per 200. Football, boxing, the World Cup,
UFC — all `sport`, including a Myanmar fighter abroad.
