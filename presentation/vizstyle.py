"""Shared chart style. Values are the dataviz reference palette (light mode),
validated with scripts/validate_palette.js:
  5-slot adjacent  -> ALL PASS (contrast WARN -> relief: direct labels + table view)
  4-slot adjacent  -> ALL PASS (same relief)
  2-slot all-pairs -> ALL PASS
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK2      = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
DEEMPH    = "#d8d7d0"          # de-emphasis gray for the emphasis form

# categorical slots, fixed order, never cycled
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
S6, S7, S8 = "#008300", "#6250d6", "#e34948"
CAT = [S1, S2, S3, S4, S5, S6, S7, S8]

# sequential blue ramp (steps 100..700)
BLUE = {100:"#cde2fb",150:"#b7d3f6",200:"#9ec5f4",250:"#86b6ef",300:"#6da7ec",
        350:"#5598e7",400:"#3987e5",450:"#2a78d6",500:"#256abf",550:"#1c5cab",
        600:"#184f95",650:"#104281",700:"#0d366b"}

CRITICAL = "#d03b3b"

def init():
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Sauce Grotesk", "Saes Grotesk", "Hanken Grotesk",
                            "Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 17, "axes.titleweight": "bold",
        "axes.labelsize": 13, "axes.labelcolor": INK2,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 12,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.3,
    })

def clean(ax, xgrid=False, ygrid=False, spines=("left", "bottom")):
    """Hairline solid recessive grid; drop the chartjunk spines."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in spines)
        if s in spines:
            ax.spines[s].set_color(BASELINE)
            ax.spines[s].set_linewidth(0.8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.8, ls="-")
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.8, ls="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

def title(ax, main, sub=None):
    # The subtitle sits in the title's pad, so the pad must grow with its line
    # count. A fixed 38 only fits one line; two-line subtitles (fig06/07/09/12)
    # were printed on top of the title.
    lines = sub.count("\n") + 1 if sub else 0
    ax.set_title(main, color=INK, loc="left", pad=22 + 17 * lines if sub else 12)
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 9), textcoords="offset points",
                    ha="left", va="bottom", fontsize=12.5, color=INK2)

def ink_on(hexfill):
    """White or ink for a label set inside a colored fill, by luminance."""
    h = hexfill.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4
    lum = .2126 * f(r) + .7152 * f(g) + .0722 * f(b)
    return "#ffffff" if lum < 0.42 else INK
