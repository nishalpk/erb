"""Formatting and plot styles shared by the analysis reports."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import REPORTS  # noqa: E402

# Reference categorical palette (fixed order; colour follows the model, never its rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
        "axes.edgecolor": BASE,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "figure.dpi": 130,
    }
)

FIGDIR = REPORTS / "figures"
TABDIR = REPORTS / "tables"


def format_number(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


def markdown_table(df: pd.DataFrame, nd=3) -> str:
    if df is None or len(df) == 0:
        return "_(no data)_\n"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            if str(c) == "p" or str(c).startswith("p_"):
                d[c] = d[c].map(lambda v: "–" if pd.isna(v) else f"{v:.2g}")
            else:
                d[c] = d[c].map(lambda v: format_number(v, nd))
    cols = list(d.columns)
    out = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n"
    for r in d.itertuples(index=False):
        out += "| " + " | ".join(str(v) for v in r) + " |\n"
    return out
