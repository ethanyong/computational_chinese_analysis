"""Shared figure style: one categorical order per desk (never cycled), thin marks, recessive
grid, a 95% interval band on every estimate, and desk identity encoded by marker as well as
colour (palette = validated reference order: blue, orange, aqua, yellow)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

DESK_COLOR = {"china": "#2a78d6", "world": "#eb6834", "singapore": "#1baf7a", "sea": "#eda100"}
DESK_MARKER = {"china": "o", "world": "s", "singapore": "^", "sea": "D"}
DESK_LABEL = {"china": "China desk", "world": "World desk", "singapore": "Singapore desk", "sea": "SE Asia desk"}
CAT_COLOR = {"hedge": "#2a78d6", "booster": "#eb6834", "obligation": "#1baf7a",
             "permission": "#eda100", "attribution": "#e87ba4"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EVENT_STYLE = dict(color="#b9b8b2", lw=0.9, zorder=0)

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 5, "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.family": ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"],  # CJK glyphs for word plots
    "axes.unicode_minus": False,
    # italic pinyin in labels is set as math text; use a real italic text face for it
    "mathtext.fontset": "custom", "mathtext.it": "Segoe UI:italic", "mathtext.rm": "Segoe UI",
})


_S2TW = None


def zh(text):
    """Display Chinese in traditional characters. The analysis itself runs on simplified text
    (jieba segments it more reliably); only labels shown to readers are converted."""
    global _S2TW
    if _S2TW is None:
        from opencc import OpenCC
        _S2TW = OpenCC("s2tw").convert
    return _S2TW(text).replace("臺", "台")   # common form, as in the paper (台灣, 台北)


def events(ax, label=True, y=None):
    """Vertical rules at the three candidate breakpoints (solid hairlines, labelled once)."""
    from config import EVENTS
    for name, x in EVENTS.items():
        ax.axvline(x, **EVENT_STYLE)
        if label:
            yy = ax.get_ylim()[1] if y is None else y
            ax.text(x + 0.08, yy, name, rotation=90, va="top", ha="left", fontsize=7.5, color=INK2)


def save(fig, name):
    """Save the figure, plus a copy for the paper (figures/paper/) without the baked-in
    "Figure N." title, because LaTeX supplies the number and caption there."""
    import os
    import re
    from config import FIGURES
    fig.savefig(f"{FIGURES}/{name}.png")
    os.makedirs(f"{FIGURES}/paper", exist_ok=True)
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        t = ax.get_title(loc="left")
        if t.startswith("Figure"):
            ax.set_title(re.sub(r"^Figure S?\d+\.\s*", "", t), loc="left", fontsize=10)
    fig.savefig(f"{FIGURES}/paper/{name}.png")
    plt.close(fig)
