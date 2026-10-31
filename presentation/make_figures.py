#!/usr/bin/env python3
"""Presentation figures for MyHumNews. Run:  .venv/bin/python presentation/make_figures.py

Numbers come from, in order of preference:
  presentation/corpus_stats.json  recomputed from data-vm/ (preprocessing funnel)
  presentation/token_stats.json   recomputed with the xlm-roberta-base tokenizer
  presentation/model_stats.json   recomputed from prediction dumps + checkpoints
  classifier/data/*.jsonl         read directly
  README.md                       ONLY where the generating script is not in the
                                  repo; each such table is marked README_* below
                                  with its line number.
"""
from __future__ import annotations
import collections, itertools, json, os, statistics as st, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, HERE)
import vizstyle as V
V.init()

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

CORPUS = json.load(open(f"{HERE}/corpus_stats.json"))
TOKENS = json.load(open(f"{HERE}/token_stats.json"))
MODELS = json.load(open(f"{HERE}/model_stats.json"))
GOLD = [json.loads(l) for l in open("classifier/data/humset_gold.jsonl")]
R2_PATH = "classifier/data/annot-thin/labels-HsuLinHtet.jsonl"
R2 = [r for r in (json.loads(l) for l in open(R2_PATH))
      if r.get("annotator") == "HsuLinHtet"]
assert len(R2) == 360, f"round 2 should be complete at 360, got {len(R2)}"
THIN = {json.loads(l)["url"]: json.loads(l)
        for l in open("classifier/data/humset_thin_task.jsonl")}

DISPLAY = {"context": "Context", "shock_event": "Shock / Event",
           "displacement": "Displacement", "casualties": "Casualties",
           "humanitarian_access": "Humanitarian access",
           "information_communication": "Information & comms", "other": "Other"}
ORDER = ["context", "shock_event", "casualties", "other", "displacement",
         "humanitarian_access", "information_communication"]
OUTLET = {"dvb": "DVB", "irrawaddy": "Irrawaddy", "bbc": "BBC",
          "myanmarnow": "Myanmar Now", "rfa": "RFA"}

# ---- README-sourced tables (generating scripts are not in the repo) ----
README_LEARNING_CURVE = {            # README.md:1477-1480
    "shock_event":  {5: .026, 10: .223, 15: .433, 20: .556, 30: .678, 45: .772},
    "context":      {5: .000, 10: .048, 15: .101, 20: .210, 30: .387, 45: .627},
    "casualties":   {5: .008, 10: .103, 15: .241, 20: .379, 30: .539, 45: .675},
    "displacement": {5: .000, 10: .055, 15: .148, 20: .343, 30: .505, 45: .627},
}
README_WEAK = [                      # README.md:1602-1608
    ("gold only", 0,      .755, .569), ("400",       400,   .759, .546),
    ("1,200",     1200,   .745, .509), ("3,000",     3000,  .729, .449),
    ("16,087",    16087,  .702, .290),
]
README_MATCHED = {                   # README.md:1744-1756 (TRUNC column)
    "context": .810, "shock_event": .861, "casualties": .733, "other": .760,
    "displacement": .472, "humanitarian_access": .000,
    "information_communication": .000,
}
README_MATCHED_MACRO = 0.727         # README.md:1754
TFIDF_FULL_MACRO = 0.755

def _num(x, _p=None):
    if x >= 1000:
        v = x / 1000
        return f"{v:,.0f}K" if abs(v - round(v)) < 1e-9 else f"{v:,.1f}K"
    return f"{x:,.0f}"
THOUS = FuncFormatter(_num)
def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------- fig 01
def fig01_pipeline():
    """Collect -> Clean -> Label -> Classify -> Retrieve, with real counts."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    g = CORPUS["global"]
    seeds = [MODELS[f"xlmr_seed{x}"]["macro_sup"] for x in (20261015, 20261016, 20261017)]
    stages = [
        ("1 · COLLECT", f"{g['read']:,}", "articles crawled",
         "5 Burmese outlets\nJan 2025 - Sep 2026", V.BLUE[450], False),
        ("2 · CLEAN", f"{g['kept']:,}", "model-ready",
         f"\u2212{g['dup_url']} duplicate URLs\n\u2212{g['too_short']} bodies <200 chars\n"
         f"\u2212{g['dup_body']} duplicate bodies", V.BLUE[450], False),
        ("3 · LABEL", "383", "gold + 360 round 2",
         "7 HumSet labels\nmulti-label, by hand\n1.0% of the corpus",
         V.BLUE[450], False),
        ("4 · CLASSIFY", f"{README_MATCHED_MACRO:.3f}", "macro-F1 (TF-IDF)",
         f"vs {st.mean(seeds):.3f} XLM-R\nmatched 512-token input\n5 of 7 labels measurable",
         V.BLUE[450], False),
        ("5 · RETRIEVE", "0.586", "Recall@10 (RAG)",
         "BGE-M3 + gazetteer\n5/5 negative refusal\n0 invalid citations",
         V.BLUE[450], False),
    ]
    fig, ax = plt.subplots(figsize=(15, 4.5))
    n = len(stages)
    w, gap = 0.172, 0.035
    for i, (head, big, small, note, col, ghost) in enumerate(stages):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch(
            (x, 0.30), w, 0.56,
            boxstyle="round,pad=0.006,rounding_size=0.018",
            facecolor=V.SURFACE if ghost else "#eef4fd",
            edgecolor=col, linewidth=1.6 if not ghost else 1.2,
            linestyle="-" if not ghost else (0, (4, 3)), zorder=3))
        ax.text(x + w / 2, 0.805, head, ha="center", va="center",
                fontsize=12, color=col, fontweight="bold", zorder=4)
        ax.text(x + w / 2, 0.665, big, ha="center", va="center",
                fontsize=24 if not ghost else 17, color=V.INK if not ghost else V.MUTED,
                fontweight="bold", zorder=4)
        ax.text(x + w / 2, 0.567, small, ha="center", va="center",
                fontsize=12, color=V.INK2, zorder=4)
        ax.text(x + w / 2, 0.435, note, ha="center", va="center",
                fontsize=11, color=V.MUTED, linespacing=1.55, zorder=4)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w + 0.004, 0.58), (x + w + gap - 0.004, 0.58),
                arrowstyle="-|>", mutation_scale=15,
                color=V.BASELINE, linewidth=1.6, zorder=2))
    ax.set_xlim(-0.012, n * (w + gap) - gap + 0.012)
    ax.set_ylim(0.16, 1.07)
    ax.axis("off")
    ax.text(0, 1.045, "The pipeline, with the counts that actually survive each stage",
            fontsize=18, fontweight="bold", color=V.INK, va="top")
    ax.text(0, 0.955,
            f"Cleaning removes only {g['read'] - g['kept']} of {g['read']:,} articles "
            f"({(g['read'] - g['kept']) / g['read']:.2%}) \u2014 the crawl was already clean. "
            "The real bottleneck is stage 3: everything downstream rests on 383 "
            "hand-labelled articles.",
            fontsize=12.5, color=V.INK2, va="top")
    save(fig, "fig01_pipeline.png")


# ---------------------------------------------------------------- fig 02
def fig02_outlet_composition():
    ps = CORPUS["per_site"]
    sites = sorted(ps, key=lambda s: -ps[s]["kept"])
    kept = [ps[s]["kept"] for s in sites]
    dropped = [ps[s]["read"] - ps[s]["kept"] for s in sites]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    y = np.arange(len(sites))[::-1]
    ax.barh(y, kept, height=.38, color=V.BLUE[450], zorder=3)
    for yy, v in zip(y, kept):
        ax.text(v + 280, yy, f"{v:,}", va="center", fontsize=13.5,
                color=V.INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([OUTLET[s] for s in sites],
                                         fontsize=13, color=V.INK2)
    ax.set_xlim(0, max(kept) * 1.16)
    ax.xaxis.set_major_formatter(THOUS)
    V.clean(ax, xgrid=True, spines=("bottom",))
    V.title(ax, "The corpus is two-thirds DVB",
            f"Model-ready articles per outlet (n = {sum(kept):,})")

    ax = axes[1]
    ax.barh(y, dropped, height=.38, color=V.DEEMPH, zorder=3)
    hi = int(np.argmax(dropped))
    ax.barh([y[hi]], [dropped[hi]], height=.38, color=V.CRITICAL, zorder=4)
    for yy, v in zip(y, dropped):
        ax.text(v + max(dropped) * .03, yy, f"{v:,}", va="center",
                fontsize=12.5, color=V.INK)
    ax.set_yticks(y); ax.set_yticklabels([OUTLET[s] for s in sites],
                                         fontsize=12, color=V.MUTED)
    ax.set_xlim(0, max(dropped) * 1.25)
    V.clean(ax, xgrid=True, spines=("bottom",))
    V.title(ax, "Cleaning losses are concentrated",
            "RFA: 80 duplicate bodies (programme boilerplate)")
    save(fig, "fig02_outlet_composition.png")


# ---------------------------------------------------------------- fig 03
def fig03_timeline():
    mbs = CORPUS["months_by_site"]
    months = sorted(CORPUS["months"])
    sites = sorted(CORPUS["per_site"], key=lambda s: -CORPUS["per_site"][s]["kept"])
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(months))
    bottom = np.zeros(len(months))
    for i, s in enumerate(sites):
        v = np.array([mbs[s].get(m, 0) for m in months], float)
        ax.bar(x, v, bottom=bottom, width=.55, color=V.CAT[i],
               label=OUTLET[s], zorder=3,
               edgecolor=V.SURFACE, linewidth=1.1)   # 2px-equivalent surface gap
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m[5:7]}/{m[2:4]}" for m in months],
                       rotation=0, ha="center", fontsize=10.5)
    ax.yaxis.set_major_formatter(THOUS)
    V.clean(ax, ygrid=True)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, -.20),
              handlelength=1.1, columnspacing=1.6)
    tot = int(bottom.sum())
    ax.annotate("Sep 2026 is a\npartial month\n(crawl ends 26 Sep)",
                xy=(len(months) - 1, bottom[-1] + 60), xytext=(-18, 150),
                textcoords="offset points", ha="right", fontsize=11,
                color=V.MUTED, linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=V.BASELINE, lw=1.2))
    V.title(ax, "Collection volume by month",
            f"{tot:,} model-ready articles, {months[0]} to {months[-1]}. "
            "Per-month values are tabulated in FIGURES.md.")
    ax.set_ylabel("Articles")
    save(fig, "fig03_corpus_timeline.png")


# ---------------------------------------------------------------- fig 04
def fig04_token_cliff():
    toks = np.array([a["tokens"] for a in TOKENS["per_article"]])
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    bins = np.arange(0, 2050, 50)
    n, edges = np.histogram(toks, bins=bins)
    mid = (edges[:-1] + edges[1:]) / 2
    cols = [V.BLUE[450] if m < TOKENS["max_len"] else V.DEEMPH for m in mid]
    ax.bar(mid, n, width=44, color=cols, zorder=3)
    ax.axvline(TOKENS["max_len"], color=V.CRITICAL, lw=2, zorder=5)
    ax.annotate(f"512-token limit\n(architectural)",
                xy=(TOKENS["max_len"], max(n) * .93),
                xytext=(TOKENS["max_len"] + 130, max(n) * .93),
                fontsize=13, color=V.CRITICAL, va="top",
                arrowprops=dict(arrowstyle="-", color=V.CRITICAL, lw=1.4))
    ax.annotate(f"{TOKENS['truncated']} of {TOKENS['n']} articles "
                f"({TOKENS['truncated_frac']:.1%}) are cut off here\n"
                f"{TOKENS['tokens_discarded']:,} of {TOKENS['tokens_total']:,} "
                f"tokens discarded ({TOKENS['tokens_discarded_frac']:.1%})",
                xy=(1180, max(n) * .55), fontsize=13, color=V.INK2, va="top")
    ax.set_xlim(0, 2050); ax.set_xlabel("XLM-R tokens per article")
    ax.set_ylabel("Articles")
    V.clean(ax, ygrid=True)
    V.title(ax, "XLM-R cannot read 60% of these articles",
            f"xlm-roberta-base tokenizer on the {TOKENS['n']} hand-labelled "
            f"articles · median {TOKENS['tokens_median']:,.0f} tokens, "
            f"{TOKENS['tokens_per_char']:.2f} tokens per character")
    save(fig, "fig04_token_cliff.png")


# ---------------------------------------------------------------- fig 05
def fig05_label_distribution():
    c = collections.Counter(l for r in GOLD for l in r["labels"])
    n = len(GOLD)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6),
                             gridspec_kw={"width_ratios": [1.75, 1]})
    ax = axes[0]
    y = np.arange(len(ORDER))[::-1]
    vals = [c[l] for l in ORDER]
    cols = [V.CRITICAL if c[l] < 20 else V.BLUE[450] for l in ORDER]
    ax.barh(y, vals, height=.42, color=cols, zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + 2.5, yy, f"{v}  ({v/n:.0%})", va="center",
                fontsize=13, color=V.INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([DISPLAY[l] for l in ORDER],
                                         fontsize=13, color=V.INK2)
    ax.set_xlim(0, max(vals) * 1.30)
    ax.axvline(20, color=V.INK2, lw=1.1, zorder=5, alpha=.5)
    ax.set_ylim(-0.6, 7.25)
    ax.annotate("20 positives = minimum for a reportable F1",
                xy=(22, 6.55), fontsize=11, color=V.INK2, va="bottom",
                annotation_clip=False)
    V.clean(ax, xgrid=True, spines=("bottom",))
    V.title(ax, "Two of seven labels are below the measurement floor",
            f"Hand-labelled gold set, n = {n} articles (red = under 20 positives)")
    ax.set_xlabel("Articles carrying the label")

    ax = axes[1]
    card = collections.Counter(len(r["labels"]) for r in GOLD)
    ks = sorted(card)
    ax.bar(ks, [card[k] for k in ks], width=.38, color=V.BLUE[450], zorder=3)
    for k in ks:
        ax.text(k, card[k] + 4, f"{card[k]}", ha="center", fontsize=13,
                color=V.INK, fontweight="bold")
    ax.set_xticks(ks); ax.set_xlabel("Labels per article")
    ax.set_ylabel("Articles"); ax.set_ylim(0, max(card.values()) * 1.18)
    V.clean(ax, ygrid=True)
    mean = sum(len(r["labels"]) for r in GOLD) / n
    V.title(ax, "It is genuinely multi-label",
            f"{sum(v for k,v in card.items() if k>1)/n:.0%} carry more than one · "
            f"mean {mean:.2f}")
    save(fig, "fig05_label_distribution.png")


# ---------------------------------------------------------------- fig 06
def fig06_cooccurrence():
    co = np.zeros((len(ORDER), len(ORDER)), int)
    idx = {l: i for i, l in enumerate(ORDER)}
    for r in GOLD:
        ls = r["labels"]
        for a in ls:
            co[idx[a]][idx[a]] = co[idx[a]][idx[a]]  # diagonal filled below
        for a, b in itertools.combinations(ls, 2):
            co[idx[a]][idx[b]] += 1
            co[idx[b]][idx[a]] += 1
    supp = collections.Counter(l for r in GOLD for l in r["labels"])
    fig, ax = plt.subplots(figsize=(8.4, 7))
    m = np.ma.masked_where(np.eye(len(ORDER), dtype=bool), co)
    vmax = m.max()
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", [V.BLUE[100], V.BLUE[250], V.BLUE[450], V.BLUE[700]])
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax)
    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor="#f2f1ec", zorder=2))
                ax.text(j, i, f"{supp[ORDER[i]]}", ha="center", va="center",
                        fontsize=12, color=V.MUTED, zorder=3)
            elif co[i][j]:
                ax.text(j, i, f"{co[i][j]}", ha="center", va="center",
                        fontsize=12.5,
                        color=V.ink_on(plt.matplotlib.colors.to_hex(
                            cmap(co[i][j] / vmax))))
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([DISPLAY[l] for l in ORDER], rotation=40, ha="right",
                       fontsize=11.5, color=V.INK2)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([DISPLAY[l] for l in ORDER], fontsize=11.5, color=V.INK2)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, len(ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ORDER), 1), minor=True)
    ax.grid(which="minor", color=V.SURFACE, lw=2)
    V.title(ax, "Which labels travel together",
            "Gold set, n = 383. Diagonal (gray) is the label's own support.\n"
            "Every one of the 85 `casualties` articles is also `shock_event`.")
    save(fig, "fig06_label_cooccurrence.png")


# ---------------------------------------------------------------- fig 07
def fig07_thin_label_fix():
    g = collections.Counter(l for r in GOLD for l in r["labels"])
    r2 = collections.Counter(l for r in R2 for l in r["labels"])
    labs = ["humanitarian_access", "information_communication"]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    y = np.arange(len(labs))[::-1].astype(float)
    base = [g[l] for l in labs]
    add = [r2[l] for l in labs]
    ax.barh(y, base, height=.30, color=V.BLUE[650], zorder=4,
            label="Gold set, random sample (383 articles)")
    ax.barh(y, add, left=np.array(base) + 1.2, height=.30, color=V.BLUE[250],
            zorder=4, label="Round 2, targeted sample (360 articles)")
    for yy, bv, av in zip(y, base, add):
        ax.text(bv / 2, yy, f"{bv}", ha="center", va="center", fontsize=12,
                color="#ffffff", fontweight="bold", zorder=6)
        ax.text(bv + av + 4, yy, f"{bv + av} total", va="center",
                fontsize=14, color=V.INK, fontweight="bold", zorder=6)
        ax.text(bv + av + 4, yy - .26, f"+{av} from round 2", va="center",
                fontsize=11.5, color=V.MUTED, zorder=6)
    # threshold rules drawn ABOVE the bars so they stay visible
    for xv, cap in ((20, "20 positives\nF1 ~ 0.2-0.6"), (45, "45 positives\nF1 ~ 0.6-0.8")):
        ax.axvline(xv, color=V.INK2, lw=1.1, zorder=5, alpha=.55)
        ax.annotate(cap, xy=(xv + 2, 1.98), fontsize=11, color=V.INK2,
                    va="top", linespacing=1.5, annotation_clip=False)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[l] for l in labs], fontsize=13.5, color=V.INK2)
    ax.set_ylim(-0.75, 2.05)
    ax.set_xlim(0, 152)
    V.clean(ax, xgrid=True, spines=("bottom",))
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, -0.40),
              handlelength=1.1, columnspacing=2.2)
    V.title(ax, "Targeted sampling moved both thin labels past the floor",
            "Round 2 was deliberately enriched by a ranker, so its hit rates "
            "(50.0% and 64.4%) describe that\nenriched sample only — they are "
            "NOT corpus prevalence. Thresholds are from the learning curve.")
    ax.set_xlabel("Positive examples now available")
    save(fig, "fig07_thin_label_fix.png")


# ---------------------------------------------------------------- fig 08
def fig08_learning_curve():
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ks = [5, 10, 15, 20, 30, 45]
    labs = ["shock_event", "context", "casualties", "displacement"]
    ends = []
    for i, lab in enumerate(labs):
        v = [README_LEARNING_CURVE[lab][k] for k in ks]
        ax.plot(ks, v, color=V.CAT[i], lw=2, marker="o", ms=7,
                mec=V.SURFACE, mew=2, zorder=3, label=DISPLAY[lab])
        ends.append([v[-1], v[-1], i, lab])
    # de-collide end labels: nudge apart and draw a leader line, never stack
    ends.sort(key=lambda e: e[0])
    MIN = .042
    for j in range(1, len(ends)):
        if ends[j][1] - ends[j - 1][1] < MIN:
            ends[j][1] = ends[j - 1][1] + MIN
    for yv, ylab, i, lab in ends:
        if abs(ylab - yv) > 1e-9:
            ax.plot([ks[-1], ks[-1] + 2.6], [yv, ylab], color=V.BASELINE,
                    lw=1, zorder=2, clip_on=False)
        ax.annotate(DISPLAY[lab], xy=(ks[-1] + 2.8, ylab), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=12.5,
                    color=V.INK2, annotation_clip=False)
    g = collections.Counter(l for r in GOLD for l in r["labels"])
    for lab, col in (("humanitarian_access", V.CRITICAL),
                     ("information_communication", V.CRITICAL)):
        ax.axvline(g[lab], color=col, lw=1.4, ls="-", zorder=2, alpha=.55)
        ax.annotate(f"{DISPLAY[lab]}\nhas {g[lab]}", xy=(g[lab], .80),
                    xytext=(4, 0), textcoords="offset points", fontsize=11.5,
                    color=col, va="top")
    ax.set_xlim(0, 56); ax.set_ylim(0, .88)
    ax.set_xlabel("Training positives available for the label")
    ax.set_ylabel("F1 for that label")
    ax.set_xticks(ks)
    V.clean(ax, ygrid=True)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.44, -.14),
              handlelength=1.3, columnspacing=2.0)
    V.title(ax, "The two zeros are a data-quantity result, not a difficulty result",
            "Each label retrained with its positives capped at k in the training "
            "folds only (test folds untouched), mean of 3 subsamples.")
    save(fig, "fig08_learning_curve.png")


# ---------------------------------------------------------------- fig 09
def fig09_model_comparison():
    seeds = [MODELS[f"xlmr_seed{x}"]["macro_sup"]
             for x in (20261015, 20261016, 20261017)]
    mean, sd = st.mean(seeds), st.stdev(seeds)
    floor = MODELS["floor"]["macro_sup"]
    fig, ax = plt.subplots(figsize=(11.5, 4.3))

    ax.axvline(floor, color=V.BASELINE, lw=1.2, zorder=2)
    ax.annotate(f"floor: always predict\nContext + Shock/Event ({floor:.3f})",
                xy=(floor + .006, 1.46), fontsize=11, color=V.MUTED,
                va="top", linespacing=1.5, annotation_clip=False)

    # TF-IDF (single point, no uncertainty measured for this arm)
    ax.plot([floor, README_MATCHED_MACRO], [1, 1], color=V.CAT[0], lw=2, zorder=3)
    ax.scatter([README_MATCHED_MACRO], [1], s=190, color=V.CAT[0], zorder=5,
               edgecolor=V.SURFACE, linewidth=2)
    ax.text(README_MATCHED_MACRO + .013, 1, f"{README_MATCHED_MACRO:.3f}",
            va="center", fontsize=15, color=V.INK, fontweight="bold")

    # XLM-R: mean, +-1 sd, and the three individual seeds
    ax.plot([floor, mean], [0, 0], color=V.CAT[1], lw=2, zorder=3)
    ax.plot([mean - sd, mean + sd], [0, 0], color=V.CAT[1], lw=5,
            alpha=.30, solid_capstyle="butt", zorder=4)
    ax.scatter(seeds, [0] * 3, s=52, facecolor=V.SURFACE, edgecolor=V.CAT[1],
               linewidth=1.8, zorder=6)
    ax.scatter([mean], [0], s=190, color=V.CAT[1], zorder=7,
               edgecolor=V.SURFACE, linewidth=2)
    ax.text(mean, .26, f"{mean:.3f}", ha="center", va="bottom", fontsize=15,
            color=V.INK, fontweight="bold")
    ax.text(mean, -.26, f"3 seeds, sd {sd:.3f}  (range "
            f"{min(seeds):.3f}\u2013{max(seeds):.3f})", ha="center", va="top",
            fontsize=11.5, color=V.MUTED)

    ax.annotate("", xy=(README_MATCHED_MACRO, .52), xytext=(mean, .52),
                arrowprops=dict(arrowstyle="<->", color=V.INK2, lw=1.3))
    ax.text((mean + README_MATCHED_MACRO) / 2, .60,
            f"gap {README_MATCHED_MACRO - mean:.3f}  =  "
            f"{(README_MATCHED_MACRO - mean) / sd:.1f}x the seed noise",
            ha="center", fontsize=12, color=V.INK2, fontweight="bold")

    ax.set_yticks([1, 0])
    ax.set_yticklabels(["TF-IDF + linear SVM\nmatched 512-token input",
                        "XLM-R fine-tuned\n512-token input"], fontsize=13)
    ax.set_ylim(-0.75, 1.75)
    ax.set_xlim(floor - .01, .85)
    ax.set_xlabel("macro-F1 over the 5 measurable labels")
    V.clean(ax, xgrid=True, spines=("bottom",))
    V.title(ax, "The linear baseline wins the matched comparison",
            "Same 383 articles, same folds, same truncated input. TF-IDF on FULL "
            "text scores 0.755,\nbut that is not the comparison: XLM-R cannot see "
            "the whole article.")
    save(fig, "fig09_model_comparison.png")


# ---------------------------------------------------------------- fig 10
def fig10_weak_supervision():
    fig, ax = plt.subplots(figsize=(10.5, 5))
    xs = list(range(len(README_WEAK)))
    v = [r[2] for r in README_WEAK]
    ax.plot(xs, v, color=V.CAT[0], lw=2, marker="o", ms=8, mec=V.SURFACE,
            mew=2, zorder=4)
    ax.axhline(v[0], color=V.BASELINE, lw=1.2, zorder=2)
    ax.annotate(f"gold only: {v[0]:.3f}", xy=(len(xs) - 1.05, v[0] + .004),
                fontsize=12, color=V.MUTED, ha="right")
    for x, val in zip(xs, v):
        ax.annotate(f"{val:.3f}", xy=(x, val), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=12.5,
                    color=V.INK, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in README_WEAK], fontsize=12.5)
    ax.set_xlabel("Weakly-labelled articles added to the 383 gold articles\n(categories, not drawn to scale)")
    ax.set_ylabel("macro-F1 (5 measurable labels)")
    ax.set_ylim(.68, .78)
    V.clean(ax, ygrid=True)
    V.title(ax, "Free weak labels made it worse, in proportion to volume",
            "Outlet section names as labels (e.g. the sport section as Other, "
            "100% precise). "
            "Dilution, not label noise: small doses are neutral, large ones hurt.")
    save(fig, "fig10_weak_supervision.png")


# ---------------------------------------------------------------- fig 11
def fig11_per_label():
    seedkeys = [f"xlmr_seed{s}" for s in (20261015, 20261016, 20261017)]
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    x = np.arange(len(ORDER))
    w = .30
    tf = [README_MATCHED[l] for l in ORDER]
    xl = [st.mean([MODELS[k]["per_label"][l]["F1"] for k in seedkeys]) for l in ORDER]
    sd = [st.stdev([MODELS[k]["per_label"][l]["F1"] for k in seedkeys]) for l in ORDER]
    ax.bar(x - w / 2, tf, width=w, color=V.CAT[0], zorder=3,
           label="TF-IDF + linear SVM (matched input)",
           edgecolor=V.SURFACE, linewidth=1.1)
    ax.bar(x + w / 2, xl, width=w, color=V.CAT[1], zorder=3,
           label="XLM-R, mean of 3 seeds", edgecolor=V.SURFACE, linewidth=1.1)
    ax.errorbar(x + w / 2, xl, yerr=sd, fmt="none", ecolor=V.INK, lw=1.4,
                capsize=5, capthick=1.4, zorder=5)
    supp = collections.Counter(l for r in GOLD for l in r["labels"])
    SHORT = {"context": "Context", "shock_event": "Shock /\nEvent",
             "displacement": "Displace-\nment", "casualties": "Casualties",
             "humanitarian_access": "Humanitarian\naccess",
             "information_communication": "Information\n& comms",
             "other": "Other"}
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SHORT[l]}\nn={supp[l]}" for l in ORDER], fontsize=11)
    ax.set_ylim(0, 1.0); ax.set_ylabel("F1")
    V.clean(ax, ygrid=True)
    ax.legend(loc="upper right")
    thin_i = [i for i, l in enumerate(ORDER) if supp[l] < 20]
    if thin_i:
        ax.axvspan(min(thin_i) - .5, max(thin_i) + .5, color="#f2f1ec", zorder=1)
        ax.annotate("below the measurement floor\n(under 20 positives)",
                    xy=((min(thin_i) + max(thin_i)) / 2, .30), ha="center",
                    va="bottom", fontsize=11, color=V.MUTED, linespacing=1.5,
                    zorder=6)
    V.title(ax, "Per label: TF-IDF leads everywhere except the thinnest labels",
            "Error bars are ±1 sd across 3 XLM-R seeds. The two rightmost labels "
            "have 16 and 6 positives — treat them as a signal of shape, not a "
            "measurement.")
    save(fig, "fig11_per_label_matched.png")


# ---------------------------------------------------------------- fig 12
def fig12_annotation_ceiling():
    """Second-annotator agreement beside model F1, on the same gold labels.

    Agreement is recomputed here from the two label files; nothing is taken
    from the README. IAA run 1 is deliberately NOT plotted -- it was served
    while the guide still named the under-represented labels, and its numbers
    measure that priming rather than annotator agreement.
    """
    gold = {r["url"]: set(r["labels"]) for r in GOLD}
    # PRE-adjudication pass: annot-iaa2/README.md says agreement must come
    # from the two independent passes. The adjudicated file had 10
    # humanitarian_access labels edited after both passes were seen, which
    # raises exact match 0.26 -> 0.36. That label is not plotted here, so the
    # bars are identical either way; only the subtitle numbers move.
    B = [json.loads(l) for l in
         open("classifier/data/annot-iaa2/labels-ZweHtetAung.preadjudication-20261017.jsonl")]
    assert len(B) == 50, len(B)

    def human_f1(lab):
        tp = sum(1 for r in B if lab in r["labels"] and lab in gold[r["url"]])
        fp = sum(1 for r in B if lab in r["labels"] and lab not in gold[r["url"]])
        fn = sum(1 for r in B if lab not in r["labels"] and lab in gold[r["url"]])
        return 0.0 if not tp else 2 * tp / (2 * tp + fp + fn)

    exact = sum(1 for r in B if set(r["labels"]) == gold[r["url"]]) / len(B)
    # MEAN PER ARTICLE, matching score_multilabel.py:199 and the README.
    # The pooled ratio is a different number (0.44) and mixing the two would
    # put a figure and the write-up quietly out of step.
    jac = st.mean(len(set(r["labels"]) & gold[r["url"]])
                  / len(set(r["labels"]) | gold[r["url"]]) for r in B)

    # Only labels pass 1 actually applied in this sample can be compared.
    # `other` is 0 by construction: the second annotator had already seen all
    # 85 pass-1 `other` articles, so no unseen sample can contain one.
    shown = [l for l in ORDER if sum(1 for r in B if l in gold[r["url"]]) >= 5]
    supp = {l: sum(1 for r in B if l in gold[r["url"]]) for l in shown}

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    x = np.arange(len(shown))
    w = .30
    hum = [human_f1(l) for l in shown]
    mod = [MODELS["tfidf_full"]["per_label"][l]["F1"] for l in shown]
    ax.bar(x - w / 2, hum, width=w, color=V.CAT[4], zorder=3,
           edgecolor=V.SURFACE, linewidth=1.1,
           label="second annotator vs the same gold labels")
    ax.bar(x + w / 2, mod, width=w, color=V.CAT[0], zorder=3,
           edgecolor=V.SURFACE, linewidth=1.1,
           label="TF-IDF model vs gold (full text, 5-fold CV)")
    for xi, (h, m) in enumerate(zip(hum, mod)):
        ax.annotate(f"{h:.2f}", (xi - w / 2, h), ha="center", va="bottom",
                    fontsize=11, color=V.INK2, xytext=(0, 3),
                    textcoords="offset points")
        ax.annotate(f"{m:.2f}", (xi + w / 2, m), ha="center", va="bottom",
                    fontsize=11, color=V.INK2, xytext=(0, 3),
                    textcoords="offset points")

    SHORT = {"context": "Context", "shock_event": "Shock /\nEvent",
             "displacement": "Displace-\nment", "casualties": "Casualties"}
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SHORT.get(l, DISPLAY[l])}\nn={supp[l]}" for l in shown],
                       fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 against the gold labels")
    V.clean(ax, ygrid=True)
    # Below the axes: inside, it covered the shaded thin-label note.
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(.5, -.24),
              handlelength=1.1, columnspacing=2.2)

    thin = [i for i, l in enumerate(shown) if supp[l] < 10]
    if thin:
        ax.axvspan(min(thin) - .5, max(thin) + .5, color="#f2f1ec", zorder=1)
        ax.annotate("only %d gold positives\nin this sample" % supp[shown[thin[0]]],
                    xy=(np.mean(thin), .70), ha="center", va="bottom",
                    fontsize=11, color=V.MUTED, linespacing=1.5, zorder=6)

    V.title(ax,
            "The headline is bounded by annotation agreement, not by the model",
            f"50 gold articles re-labelled blind by a second annotator: exact "
            f"set match {exact:.2f}, Jaccard {jac:.2f}. The model was fitted to "
            f"these labels, so it has\nlearned one annotator's conventions that "
            f"a second annotator does not share \u2014 this is a ceiling, not a "
            f"model beating a human.")
    save(fig, "fig12_annotation_ceiling.png")


if __name__ == "__main__":
    for f in (fig01_pipeline, fig02_outlet_composition, fig03_timeline,
              fig04_token_cliff, fig05_label_distribution, fig06_cooccurrence,
              fig07_thin_label_fix, fig08_learning_curve, fig09_model_comparison,
              fig10_weak_supervision, fig11_per_label,
              fig12_annotation_ceiling):
        f()
    print("\nall figures written to presentation/figures/")
