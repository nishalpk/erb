"""Slide-sized figures for the deck (larger type, BITS palette). Reads ERB2 outputs; writes PNGs to argv[2]."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "reports" / "tables"
NAVY, SKY, ORANGE, RED, GREY, INK = "#101141", "#76C2E5", "#E39B00", "#E31C24", "#8A8A8A", "#1A1A1A"
MODEL = {"smolvlm2-2.2b": ("SmolVLM2-2.2B", ORANGE), "llava-ov-0.5b": ("LLaVA-OV-0.5B", RED),
         "qwen2.5-vl-3b": ("Qwen2.5-VL-3B", "#1F6FB5"), "gemini-3.5-flash-lite": ("Gemini Flash-Lite", NAVY)}
plt.rcParams.update({"font.family": "Calibri", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#B0B0B0", "axes.labelcolor": INK, "xtick.color": "#555555", "ytick.color": "#555555",
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200, "legend.frameon": False})

# ---------------------------------------------------------------- Bayesian posteriors (2 x 4)
rr = pd.read_csv(TAB / "bayes_rr.csv")
LV = [("neutral", "neutral", GREY), ("contra_1", "hedged", "#9CC9EC"), ("contra_2", "plain", "#4F95D6"),
      ("contra_3", "\"definitely\"", "#1F5FA8"), ("contra_4", "\"verified\"", NAVY)]
order = ["smolvlm2-2.2b", "qwen2.5-vl-3b", "llava-ov-0.5b"]
fig, axs = plt.subplots(2, 3, figsize=(10, 4.6), sharex=True)
x = np.linspace(0, 1, 800)
for j, m in enumerate(order):
    for i, ds in enumerate(["real", "synthetic"]):
        ax = axs[i, j]
        g = rr[(rr.model == m) & (rr.dataset == ds)].set_index("condition")
        for c, lab, col in LV:
            if c in g.index:
                y = stats.beta.pdf(x, g.loc[c, "alpha"], g.loc[c, "beta"])
                y = y / y.max()
                ax.plot(x, y, color=col, lw=2.2, label=lab)
                ax.fill_between(x, 0, y, color=col, alpha=0.12)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{MODEL[m][0]} · {ds} (n={int(g.n.iloc[0])})", fontsize=12, color=INK)
        if i == 1:
            ax.set_xlabel("reassignment rate θ")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, fontsize=12, title="false-caption confidence →", title_fontsize=11,
           bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(OUT / "bayes.png")
plt.close(fig)

# ---------------------------------------------------------------- PCA 2 x 2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
z = np.load(ROOT / "results" / "probe_resid_qwen2.5-vl-3b.npz")
pca_tab = pd.read_csv(TAB / "pca_probe.csv")
layer = int(pca_tab.layer.iloc[0])
A, C, y, fl, dsv = z["Xn"][:, layer].astype(np.float32), z["Xc"][:, layer].astype(np.float32), z["y"], z["flipped"].astype(bool), z["dataset"]
sc = StandardScaler().fit(A)
pca = PCA(100, random_state=0).fit(sc.transform(A))
Pn, Pc = pca.transform(sc.transform(A)), pca.transform(sc.transform(C))
cum = np.cumsum(pca.explained_variance_ratio_)
fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.0))
for ax, P, t in [(axs[0, 0], Pn, "No caption"), (axs[0, 1], Pc, "False caption (contra_3)")]:
    for lab, col, nm in [(1, "#1F6FB5", "true actor = A"), (0, ORANGE, "true actor = B")]:
        for ds, mk in (("real", "o"), ("synthetic", "^")):
            msk = (y == lab) & (dsv == ds)
            ax.scatter(P[msk, 0], P[msk, 1], s=14, color=col, marker=mk, alpha=0.65, edgecolors="none",
                       label=nm if ds == "real" else None)
    if P is Pc:
        ax.scatter(P[fl, 0], P[fl, 1], s=40, facecolors="none", edgecolors=INK, lw=0.7, label="answer flipped")
    ax.set_title(f"{t}: layer {layer}", fontsize=11)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})\n(circles = real, triangles = synthetic)", fontsize=9)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.legend(fontsize=8, loc="center", markerscale=1.1, handletextpad=0.2)
ax = axs[1, 0]
ax.plot(np.arange(1, 101), cum, color=NAVY, lw=2.4)
for thr in (0.5, 0.9):
    k = int(np.searchsorted(cum, thr) + 1)
    ax.axhline(thr, color=GREY, ls="--", lw=0.9)
    ax.annotate(f"{thr:.0%} variance at k = {k}", (k, thr), xytext=(6, -16), textcoords="offset points", fontsize=9)
ax.set_xlabel("principal components k (of 2,048 dims)")
ax.set_ylabel("cumulative explained variance")
ax.set_title("Scree curve", fontsize=11)
ax = axs[1, 1]
xs = np.arange(len(pca_tab))
ax.plot(xs, pca_tab.acc_none, "-o", color="#1F6FB5", lw=2, ms=5, label="no caption")
ax.plot(xs, pca_tab.acc_contra3_flipped, "-o", color=RED, lw=2, ms=5, label="false caption, answer flipped")
ax.axhline(0.5, color=GREY, ls="--", lw=0.9)
ax.text(len(xs) - 1, 0.52, "chance", ha="right", fontsize=10, color=GREY)
ax.set_xticks(xs, pca_tab.k)
ax.set_ylim(0, 1)
ax.set_xlabel("PCA components kept before the probe")
ax.set_ylabel("probe accuracy (true actor)")
ax.set_title("Logistic probe on PCA features", fontsize=11)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "pca.png")
plt.close(fig)

# ---------------------------------------------------------------- patching heatmap
rec = pd.read_csv(TAB / "patch_recovery.csv")
M = rec[["entity", "caption_rest", "question", "final"]].to_numpy().T
fig, ax = plt.subplots(figsize=(4.6, 2.3))
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list("b", ["#FFFFFF", "#76C2E5", NAVY])
im = ax.imshow(np.clip(M, 0, 1), aspect="auto", cmap=cm, vmin=0, vmax=1)
ax.set_yticks(range(4), ["caption \"B\"", "rest of caption", "question", "final token"], fontsize=9)
ax.set_xlabel("decoder layer")
ax.set_title("Patching: where the caption's answer lives", fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.04)
cb.set_label("effect recovered", fontsize=8)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "patching.png")
plt.close(fig)
print("ok", list(OUT.iterdir()))
