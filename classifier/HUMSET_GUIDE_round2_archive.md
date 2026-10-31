# ARCHIVED — round 2 task guidance (completed 2026-10-16)

Round 2 finished at 360/360. This section was removed from the live
`HUMSET_GUIDE.md` because it names the two under-represented labels and
states how many more examples they need. Served during IAA run 1, it primed
the second annotator: +23 `humanitarian_access` and +3
`information_communication` with zero removals on either. See
`classifier/data/annot-iaa/README.md`.

Restore it only for a collection round that is genuinely targeted, never for
a blind or agreement pass.

---

# ROUND 2 — the targeted set (360 articles)

**This set is deliberately NOT a random sample.** It was selected to be rich in
the two labels that have too few examples to learn from:

| label | examples so far | needed |
|---|---|---|
| `humanitarian_access` | 16 | ~45 |
| `information_communication` | 6 | ~45 |

Everything else in the corpus already has enough. These two have so few that the
classifier scores **0.000** on them — it cannot learn a category from six
examples. This round exists to fix exactly that.

## What that means for you

**Label what is actually there.** Do not try to find these two labels. Many
articles here will turn out to be `shock_event`, `context` or `casualties`, and
that is a correct and useful answer. A biased sample labelled honestly is still
good training data; a sample labelled to match an expectation is worthless.

**The list is ordered best-guess-first.** Article 1 is the one most likely to
carry a thin label; article 360 is the least. You can stop whenever you like
and everything you have done still counts. Doing the first 150 is genuinely
useful on its own.

## The two labels this round is about

**`humanitarian_access`** — anything about aid *reaching people or failing to*:
aid being delivered, blocked, restricted or cut; a road or border closed to
relief; medical care unavailable; an agency denied entry; funding withdrawn.

> Examples from the data that ARE `humanitarian_access`:
> - AA delivering food to Rohingya in Pauktaw
> - WFP starting emergency logistics for flood response
> - India capping the border crossing at 100 people a day
> - MSF reporting obstacles to TB treatment
> - Flood victims in Tanintharyi needing drinking water and medicine
> - US health aid being halted

Note that most of these are **also** `shock_event` or `displacement`. Apply both.

**`information_communication`** — the flow of information itself: internet or
phone shutdowns, journalists detained or killed, press freedom, censorship,
media bans, propaganda, platform takedowns.

> Examples from the data that ARE `information_communication`:
> - a journalist awarded a press-freedom prize while in prison
> - internet cut in the areas hit hardest by airstrikes
> - 17 journalists still detained
> - Telegram sued for not removing violent posts

It is **not** "news about anything". It is news about *communication being
controlled or broken*.

## Everything else is unchanged

Same seven labels, same rules: `other` is still exclusive, `casualties` still
requires `shock_event`, Myanmar politics and economy are still `context` and not
`other`. Press `?` for the definitions.
