"""Estimate Bayesian reassignment rates and analyze saved hidden states with PCA.

1. Bayesian reassignment rate (Beta-Binomial). Each rebinding outcome is Bernoulli(theta); with a
   Beta(1, 1) prior and k rebinds out of n competent items the posterior is Beta(1 + k, 1 + n - k).
   Reports posterior mean, 95% credible interval, and Monte-Carlo posterior probabilities
   P(theta_contra > theta_neutral) and P(theta_contra_1 < theta_contra_2 < theta_contra_4) (dose-response).
   -> reports/tables/bayes_rr.csv, bayes_effects.csv, figures/fig9_bayes.png
2. PCA of the model's hidden states (needs results/probe_resid_<model>.npz from probe_model.py).
   Projects last-token residuals onto the top principal components, shows how much variance a few
   components keep, and re-runs the logistic probe on the first k components (k = 1 .. full) to show
   how much of the "who does it" signal survives dimensionality reduction.
   -> reports/tables/pca_probe.csv, figures/fig10_pca.png

    python src/bayes_and_pca.py
Writes reports/results_extras.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import REPORTS, ROOT
from reporting import FIGDIR as FIG, TABDIR as TAB, INK, INK2, MUTED, SEQ, SERIES, markdown_table, plt

LEVELS = ["neutral", "contra_1", "contra_2", "contra_3", "contra_4"]
LCOL = {"neutral": MUTED, "contra_1": SEQ[1], "contra_2": SEQ[2], "contra_3": SEQ[4], "contra_4": SEQ[6]}
rng = np.random.default_rng(2026)



def estimate_reassignment_posteriors(rr: pd.DataFrame, a0=1.0, b0=1.0, draws=40000):
    from scipy import stats

    rr = rr[rr.condition.isin(LEVELS)].copy()
    rr["alpha"], rr["beta"] = a0 + rr.k, b0 + rr.n - rr.k
    rr["post_mean"] = rr.alpha / (rr.alpha + rr.beta)
    rr["cri_lo"] = stats.beta.ppf(0.025, rr.alpha, rr.beta)
    rr["cri_hi"] = stats.beta.ppf(0.975, rr.alpha, rr.beta)
    eff = []
    for (m, ds), g in rr.groupby(["model", "dataset"]):
        g = g.set_index("condition")
        smp = {c: rng.beta(g.loc[c, "alpha"], g.loc[c, "beta"], draws) for c in g.index}
        row = {"model": m, "dataset": ds, "n": int(g.n.iloc[0])}
        for c in ("contra_1", "contra_2", "contra_4"):
            if c in smp and "neutral" in smp:
                row[f"P({c}>neutral)"] = float((smp[c] > smp["neutral"]).mean())
        if all(c in smp for c in ("contra_1", "contra_2", "contra_4")):
            row["P(c1<c2<c4)"] = float(
                ((smp["contra_1"] < smp["contra_2"]) & (smp["contra_2"] < smp["contra_4"])).mean()
            )
        if "contra_4" in smp and "neutral" in smp:
            d = smp["contra_4"] - smp["neutral"]
            row["delta_c4_mean"], row["delta_c4_lo"], row["delta_c4_hi"] = (
                float(d.mean()),
                *np.quantile(d, [0.025, 0.975]).tolist(),
            )
        eff.append(row)
    return rr, pd.DataFrame(eff)


def plot_reassignment_posteriors(rr):
    from scipy import stats

    cells = rr[["model", "dataset"]].drop_duplicates().sort_values(["model", "dataset"]).values.tolist()
    ncol = 4
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.9 * nrow), sharex=True, squeeze=False)
    x = np.linspace(0, 1, 600)
    for ax, (m, ds) in zip(axs.flat, cells):
        g = rr[(rr.model == m) & (rr.dataset == ds)].set_index("condition")
        for c in LEVELS:
            if c not in g.index:
                continue
            y = stats.beta.pdf(x, g.loc[c, "alpha"], g.loc[c, "beta"])
            ax.plot(x, y / y.max(), color=LCOL[c], lw=2, label=c)
            ax.fill_between(x, 0, y / y.max(), color=LCOL[c], alpha=0.12)
        ax.set_title(f"{m} / {ds} (n={int(g.n.iloc[0])})", fontsize=9)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    for ax in axs.flat[len(cells) :]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("reassignment rate θ")
    from matplotlib.lines import Line2D

    fig.legend(
        [Line2D([], [], color=LCOL[c], lw=2.5) for c in LEVELS],
        LEVELS,
        loc="upper center",
        ncol=len(LEVELS),
        bbox_to_anchor=(0.5, 0.95),
        fontsize=9,
    )
    fig.suptitle(
        "Fig 9. Posterior of the reassignment rate, Beta(1,1) prior + Binomial likelihood (density scaled to max 1)",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "fig9_bayes.png")
    plt.close(fig)



def probe_principal_components(npz_path: Path, probe_csv: Path):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    z = np.load(npz_path, allow_pickle=False)
    Xn, Xc, y, flipped, groups = (
        z["Xn"].astype(np.float32),
        z["Xc"].astype(np.float32),
        z["y"],
        z["flipped"].astype(bool),
        z["item_id"],
    )
    nL = Xn.shape[1]
    if probe_csv.exists():
        pb = pd.read_csv(probe_csv)
        mid = pb.iloc[nL // 4 : 3 * nL // 4]
        layer = int(mid.loc[mid.acc_none.idxmax(), "layer"])
    else:
        layer = nL // 2
    A, C = Xn[:, layer], Xc[:, layer]
    sc = StandardScaler().fit(A)
    pca = PCA(n_components=min(100, len(A) - 1), random_state=0).fit(sc.transform(A))
    Pn, Pc = pca.transform(sc.transform(A)), pca.transform(sc.transform(C))
    cum = np.cumsum(pca.explained_variance_ratio_)

    gkf = GroupKFold(n_splits=5)
    ks = [k for k in (1, 2, 3, 5, 10, 20, 50, 100) if k <= pca.n_components_] + ["full"]
    rows = []
    for k in ks:
        pn, pc = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in gkf.split(A, y, groups=groups):
            steps = (
                [StandardScaler()]
                + ([PCA(n_components=k, random_state=0)] if k != "full" else [])
                + [LogisticRegression(C=0.1, max_iter=3000)]
            )
            clf = make_pipeline(*steps).fit(A[tr], y[tr])
            pn[te], pc[te] = clf.predict(A[te]), clf.predict(C[te])
        ok_c = pc == y
        rows.append(
            {
                "layer": layer,
                "k": str(k),
                "dims": A.shape[1] if k == "full" else k,
                "acc_none": float((pn == y).mean()),
                "acc_contra3": float(ok_c.mean()),
                "acc_contra3_flipped": float(ok_c[flipped].mean()) if flipped.any() else np.nan,
                "n_flipped": int(flipped.sum()),
                "var_explained": float(cum[k - 1]) if k != "full" else 1.0,
            }
        )
    res = pd.DataFrame(rows)

    fig, axs = plt.subplots(1, 4, figsize=(18, 4.3))
    cA, cB = SERIES[0], SERIES[1]
    dsv = z["dataset"]
    for ax, P, title in [
        (axs[0], Pn, "no caption (none)"),
        (axs[1], Pc, "contradictory caption (contra_3)"),
    ]:
        for lab, col, nm in [(1, cA, "true actor = A"), (0, cB, "true actor = B")]:
            for ds, mk in (("real", "o"), ("synthetic", "^")):
                msk = (y == lab) & (dsv == ds)
                ax.scatter(
                    P[msk, 0],
                    P[msk, 1],
                    s=12,
                    color=col,
                    alpha=0.6,
                    marker=mk,
                    edgecolors="none",
                    label=nm if ds == "real" else None,
                )
        if P is Pc and flipped.any():
            ax.scatter(
                P[flipped, 0],
                P[flipped, 1],
                s=36,
                facecolors="none",
                edgecolors=INK,
                lw=0.7,
                label="output flipped",
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%}): ● real vs ▲ synthetic")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        ax.set_title(f"Layer {layer}: {title}")
        ax.legend(fontsize=7, markerscale=1.2)
    ax = axs[2]
    ax.plot(np.arange(1, len(cum) + 1), cum, color=SERIES[2], lw=2)
    for thr in (0.5, 0.9):
        kk = int(np.searchsorted(cum, thr) + 1) if cum[-1] >= thr else None
        if kk:
            ax.axhline(thr, color=MUTED, ls="--", lw=0.8)
            ax.annotate(
                f"{thr:.0%} at k={kk}",
                (kk, thr),
                xytext=(4, -12),
                textcoords="offset points",
                fontsize=8,
                color=INK2,
            )
    ax.set_xlabel("number of components k")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title(f"Scree: {A.shape[1]}-dim residual stream")
    ax = axs[3]
    xs = np.arange(len(res))
    ax.plot(xs, res.acc_none, "-o", color=SERIES[0], ms=4, label="test on none")
    ax.plot(xs, res.acc_contra3, "-o", color=SERIES[1], ms=4, label="test on contra_3")
    if res.acc_contra3_flipped.notna().any():
        ax.plot(
            xs,
            res.acc_contra3_flipped,
            "-o",
            color=SERIES[7],
            ms=4,
            label=f"contra_3, flipped (n={int(flipped.sum())})",
        )
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_xticks(xs, res.k)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("PCA components kept before logistic probe")
    ax.set_ylabel("probe accuracy (true actor)")
    ax.set_title("Probe on PCA-reduced features")
    ax.legend(fontsize=7)
    fig.suptitle(
        f"Fig 10. PCA of last-token hidden states ({npz_path.stem.replace('probe_resid_', '')}, n={len(y)} items)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig10_pca.png")
    plt.close(fig)
    return res, layer, cum


def main():
    md = ["# Extra analyses (course-aligned)", "", "Generated by `src/bayes_and_pca.py`.", ""]
    rr = pd.read_csv(TAB / "rr.csv")
    post, eff = estimate_reassignment_posteriors(rr)
    post.drop(columns=["ERBS", "mean_logit_diff", "ties"], errors="ignore").to_csv(
        TAB / "bayes_rr.csv", index=False
    )
    eff.to_csv(TAB / "bayes_effects.csv", index=False)
    plot_reassignment_posteriors(post)
    md += [
        "## 1. Bayesian reassignment rate (Beta-Binomial)",
        "",
        "Prior θ ~ Beta(1, 1) (uniform); likelihood k ~ Binomial(n, θ) over competent items; posterior "
        "θ | k ~ Beta(1 + k, 1 + n − k). Posterior probabilities from 40,000 Monte-Carlo draws.",
        "",
        markdown_table(
            post[["model", "dataset", "condition", "n", "k", "RR", "post_mean", "cri_lo", "cri_hi"]]
        ),
        "Posterior effect probabilities:",
        "",
        markdown_table(eff),
        "![](figures/fig9_bayes.png)",
        "",
    ]
    # refresh the H5 verdict from the saved probe table (same rule as probe_model.py, analysis.h5_verdict)
    pj, pc = TAB / "probe_summary.json", TAB / "probe_by_layer.csv"
    if pc.exists():
        import json
        from analysis import h5_verdict

        s = json.loads(pj.read_text()) if pj.exists() else {}
        model = s.get("model") or next(
            (p.stem.replace("probe_resid_", "") for p in (ROOT / "results").glob("probe_resid_*.npz")), "?"
        )
        n_items = s.get("n_items") or len(pd.read_csv(TAB / "probe_items.csv"))
        s.update(
            {"model": model, "n_items": n_items, "verdict": h5_verdict(pd.read_csv(pc), model, n_items)}
        )
        pj.write_text(json.dumps(s, indent=2))
        md += ["## H5 verdict (refreshed)", "", s["verdict"], ""]
    npzs = sorted((ROOT / "results").glob("probe_resid_*.npz"))
    if npzs:
        res, layer, cum = probe_principal_components(npzs[0], TAB / "probe_by_layer.csv")
        res.to_csv(TAB / "pca_probe.csv", index=False)
        k90 = int(np.searchsorted(cum, 0.9) + 1) if cum[-1] >= 0.9 else f">{len(cum)}"
        md += [
            "## 2. PCA of hidden states + probe on reduced features",
            "",
            f"Layer {layer} (best mid-layer probe on `none`). Standardised last-token residuals; PCA fit on `none`. "
            f"{k90} components keep 90% of the variance.",
            "",
            markdown_table(res),
            "![](figures/fig10_pca.png)",
            "",
        ]
    else:
        md += [
            "## 2. PCA",
            "",
            "_Not run: results/probe_resid_*.npz missing (run src/probe_model.py)._",
            "",
        ]
    (REPORTS / "results_extras.md").write_text("\n".join(md), encoding="utf-8")
    print(eff.to_string(index=False))
    print("wrote", REPORTS / "results_extras.md")


if __name__ == "__main__":
    main()
