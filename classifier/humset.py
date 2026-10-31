#!/usr/bin/env python3
"""
HumSet-based humanitarian label set for MyHumNews. MULTI-LABEL.

The project's actual classification target (BDAPP_Topic_Proposal.pptx): HumSet's
1D pillars with COVID-19 dropped and `other` added. An article carries as many
labels as apply.

This replaces `taxonomy.py` as the TARGET. taxonomy.py is kept, not deleted:
its 10 general news topics are a measured TF-IDF baseline and remain the
comparison point for the general-topic benchmark (SIB-200) in the proposal.

WHY MULTI-LABEL CHANGES MORE THAN IT LOOKS:
  - an airstrike that kills civilians and empties a village is Shock/Event AND
    Casualties AND Displacement. Forcing one label discards most of the signal.
  - metrics become per-label precision/recall/F1 with macro-F1 over labels,
    which is what slide 5 of the proposal asks for. Plain accuracy is close to
    meaningless here, and exact-match ("did we get the whole set right") is a
    much harsher number that should be reported alongside.
  - the model becomes one-vs-rest: seven independent binary decisions.

`other` IS EXCLUSIVE. It means "no humanitarian content at all" -- a football
result, a celebrity story. If `other` applies, nothing else does, and the
labelling tools enforce that.
"""

from __future__ import annotations

LABELS = [
    "context",
    "shock_event",
    "displacement",
    "casualties",
    "humanitarian_access",
    "information_communication",
    "other",
]

EXCLUSIVE = "other"

# Short labels for the UI, keyed to the digits the labelling tools use.
DISPLAY = {
    "context":                   "Context",
    "shock_event":               "Shock / Event",
    "displacement":              "Displacement",
    "casualties":                "Casualties",
    "humanitarian_access":       "Humanitarian Access",
    "information_communication": "Information & Comms",
    "other":                     "Other (no humanitarian content)",
}

# One-line decision rules. Deliberately written for Burmese news about Myanmar,
# not copied from HumSet's generic phrasing, because the generic wording leaves
# `context` so broad it swallows everything.
DEFINITION = {
    "context":
        "The surrounding situation rather than an event: politics, economy, "
        "demography, law and policy, security and stability, socio-cultural "
        "background. Analysis and explainers usually land here.",
    "shock_event":
        "The hazard or incident itself and its characteristics -- an "
        "airstrike, an earthquake, a flood, a raid, a battle. What happened.",
    "displacement":
        "People moving or unable to move: fleeing, IDPs, refugees, camps, "
        "returns, push and pull factors, numbers displaced.",
    "casualties":
        "Dead, injured or missing people. Counts, or named victims.",
    "humanitarian_access":
        "Aid reaching people or failing to: blocked roads, denied permission, "
        "besieged areas, aid suspension, relief operations, access "
        "constraints, who is cut off.",
    "information_communication":
        "Information itself as the subject: internet or phone shutdowns, "
        "censorship, press freedom, journalists detained, rumour and "
        "misinformation, how people get news.",
    "other":
        "No humanitarian relevance at all -- sport, entertainment, celebrity, "
        "routine business. EXCLUSIVE: if this applies, nothing else does.",
}

# Guidance for the pairs that actually get confused, from reading the corpus.
NOTES = """
shock_event vs casualties
    They co-occur constantly and that is correct. "Airstrike kills 3" is BOTH:
    the airstrike is the event, the 3 deaths are casualties. Do not choose.

context vs everything
    `context` is for the backdrop, not the incident. A battle is shock_event.
    An analysis of why the offensive is happening is context. If an article
    reports something that happened, it is probably not context alone.

displacement vs humanitarian_access
    Displacement is people moving. Access is aid moving (or not). "20,000 flee
    to the border" is displacement. "Aid convoys blocked from reaching them" is
    access. An article often has both.

information_communication
    Only when information is the SUBJECT. A story about an internet shutdown
    qualifies; a story that merely quotes a Facebook post does not.

other
    Use it. Roughly a third of DVB's output is sport, entertainment and foreign
    general news with no humanitarian angle, and the proposal needs `other` to
    keep those out of the RAG index. That is a real function, not a dustbin.
"""


def validate(labels) -> list[str]:
    """Normalise and check a label set. Raises on anything invalid."""
    out = []
    for l in labels or []:
        if l not in LABELS:
            raise ValueError(f"unknown label {l!r}")
        if l not in out:
            out.append(l)
    if EXCLUSIVE in out and len(out) > 1:
        raise ValueError(f"{EXCLUSIVE!r} is exclusive, got {out}")
    return [l for l in LABELS if l in out]        # canonical order


# ---------------------------------------------------------------- outlets
#
# The label stores were written by two different generations of script and
# disagree on how an outlet is spelled: gold and everything derived from the
# corpus use the short form, while the round-2 task and its labels carry the
# crawler's directory names (`dvb_burmese`). Anything that groups by outlet
# across both -- the per-outlet report in train_humset.py, for one -- would
# otherwise split DVB into two outlets with half the articles each.
#
# The short form is canonical: it is what gold, the corpus statistics and the
# presentation already use. Normalise on READ rather than rewriting the served
# task files, which are the record of what each annotator was actually shown.
SITES = {
    "dvb_burmese": "dvb",
    "rfa_burmese": "rfa",
    "bbc_burmese": "bbc",
    "irrawaddy_burmese": "irrawaddy",
    "myanmar_now_burmese": "myanmarnow",
}
OUTLET_DISPLAY = {
    "dvb": "DVB", "rfa": "RFA", "bbc": "BBC",
    "irrawaddy": "Irrawaddy", "myanmarnow": "Myanmar Now",
}


def canon_site(site: str | None) -> str | None:
    """Canonical short outlet name. Idempotent; unknown values pass through.

    Passing unknown values through rather than raising is deliberate: a new
    outlet should show up in a report as itself, not crash the run.
    """
    if site is None:
        return None
    return SITES.get(site, site)


def fingerprint() -> str:
    """Stable hash of the label set, same role as taxonomy.fingerprint()."""
    import hashlib
    payload = "\u0000".join(["|".join(LABELS), EXCLUSIVE])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    print(f"{len(LABELS)} labels, multi-label, fingerprint {fingerprint()}")
    for i, l in enumerate(LABELS):
        key = str(i + 1) if i < 6 else "0"
        print(f"\n  [{key}] {DISPLAY[l]}")
        print(f"      {DEFINITION[l]}")
    print(NOTES)
