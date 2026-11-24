"""Analyze saved model results and write the benchmark report, tables, and figures."""

from __future__ import annotations

import argparse
import html
import json
import sys

import numpy as np
import pandas as pd

import analysis
from common import PROC, REPORTS, ROOT
from reporting import (
    FIGDIR,
    GRID,
    INK,
    INK2,
    MUTED,
    SEQ,
    SERIES,
    SURF,
    TABDIR,
    format_number,
    markdown_table,
    plt,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def model_colors(models):
    return {m: SERIES[i % len(SERIES)] for i, m in enumerate(sorted(models))}



def fig_funnel():
    p = PROC / "real" / "funnel.csv"
    if not p.exists():
        return None
    f = pd.read_csv(p)
    srcs = [c for c in f.columns if c not in ("stage", "total")]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(f) + 1.2))
    y = np.arange(len(f))[::-1]
    left = np.zeros(len(f))
    for i, s in enumerate(srcs):
        ax.barh(y, f[s], left=left, color=SERIES[i], height=0.62, label=s, edgecolor=SURF, linewidth=2)
        left += f[s].to_numpy()
    for yy, t in zip(y, f["total"]):
        ax.text(t * 1.08 if t > 0 else 1, yy, f"{t:,}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, f["stage"])
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("images surviving (log scale)")
    ax.set_title("Fig 1. Real-subset funnel: Contradiction Validity Conditions")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_funnel.png")
    plt.close(fig)
    return f


def fig_eda(items):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    real = items[items.dataset == "real"]
    ax = axs[0, 0]
    vc = items.groupby(["dataset", "verb_key"]).size().reset_index(name="n").sort_values("n")
    labels = vc.dataset.str[:5] + " · " + vc.verb_key
    ax.barh(
        range(len(vc)),
        vc.n,
        color=[SERIES[0] if d == "real" else SERIES[2] for d in vc.dataset],
        height=0.7,
    )
    ax.set_yticks(range(len(vc)), labels, fontsize=7.5)
    ax.set_title("Items by action (blue: real, aqua: synthetic)")
    ax.grid(axis="y", visible=False)
    ax = axs[0, 1]
    for i, (ds, g) in enumerate(items.groupby("dataset")):
        ax.hist(g.min_person_area, bins=30, alpha=0.75, color=SERIES[0 if ds == "real" else 2], label=ds)
    ax.set_xlabel("smaller person box (fraction of image)")
    ax.set_title("Person size")
    ax.legend()
    ax = axs[1, 0]
    for i, (ds, g) in enumerate(items.groupby("dataset")):
        ax.hist(g.person_dist, bins=30, alpha=0.75, color=SERIES[0 if ds == "real" else 2], label=ds)
    ax.set_xlabel("distance between person centres (fraction of image)")
    ax.set_title("Inter-person distance")
    ax.legend()
    ax = axs[1, 1]
    bal = items.groupby(["dataset", "actor_side", "actor_letter"]).size().unstack(fill_value=0)
    x = np.arange(len(bal))
    ax.bar(x - 0.18, bal.get("A", 0), 0.36, color=SERIES[0], label="actor = A")
    ax.bar(x + 0.18, bal.get("B", 0), 0.36, color=SERIES[1], label="actor = B")
    ax.set_xticks(x, [f"{d}\nactor {s}" for d, s in bal.index], fontsize=8)
    ax.set_title("Counterbalancing: actor side x marker letter")
    ax.legend()
    fig.suptitle(
        "Fig 2. Dataset EDA" + (f"  (real n={len(real)}, synthetic n={len(items) - len(real)})"),
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_eda.png")
    plt.close(fig)


def fig_capability(cap, colors):
    dss = sorted(cap.dataset.unique())
    tasks = list(dict.fromkeys(cap.task))
    fig, axs = plt.subplots(1, len(dss), figsize=(6.2 * len(dss), 4.6), squeeze=False)
    models = sorted(cap.model.unique())
    w = 0.8 / len(models)
    for ax, ds in zip(axs[0], dss):
        for i, m in enumerate(models):
            g = cap[(cap.model == m) & (cap.dataset == ds)].set_index("task").reindex(tasks)
            x = np.arange(len(tasks)) + (i - (len(models) - 1) / 2) * w
            ax.bar(x, g.acc, w * 0.92, color=colors[m], label=m)
            ax.errorbar(
                x, g.acc, yerr=[g.acc - g.lo, g.hi - g.acc], fmt="none", ecolor=INK2, lw=0.8, capsize=2
            )
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(np.arange(len(tasks)), [t.replace(" ", "\n", 2) for t in tasks], fontsize=7.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy (95% Wilson CI)")
        ax.set_title(ds)
        ax.grid(axis="x", visible=False)
    axs[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Fig 3. Capability panel (dashed = chance)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_capability.png")
    plt.close(fig)


def plot_dose_response(rr, colors):
    dss = sorted(rr.dataset.unique())
    fig, axs = plt.subplots(1, len(dss), figsize=(max(8, 6.2 * len(dss)), 4.6), squeeze=False, sharey=True)
    for ax, ds in zip(axs[0], dss):
        for m, g in rr[rr.dataset == ds].groupby("model", observed=True):
            g = g.set_index("condition").reindex(analysis.COND_ORDER).dropna(subset=["RR"])
            x = [analysis.COND_ORDER.index(c) for c in g.index]
            ax.fill_between(x, g.lo, g.hi, color=colors[m], alpha=0.12, lw=0)
            ax.plot(x, g.RR, "-o", color=colors[m], lw=2, ms=5, label=m)
            ax.annotate(
                m,
                (x[-1], g.RR.iloc[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color=INK2,
                va="center",
            )
        ax.axvspan(2.5, 6.5, color=GRID, alpha=0.35, lw=0)
        ax.set_xticks(
            range(len(analysis.COND_ORDER)),
            ["none", "neutral", "consistent", "hedged", "plain", "assertive", "authorit.\n+repeat"],
            fontsize=8,
        )
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{ds}")
        ax.set_ylabel("Reassignment Rate (names distractor)\non competence set S_m, 95% CI")
    axs[0, -1].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Fig 4. Dose-response: confidence of the false caption vs. rebinding\n(shaded = contradictory captions)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_dose_response.png")
    plt.close(fig)


def fig_taxonomy(tax):
    order = [
        "Stable",
        "Rebound, perception intact",
        "Rebound, perception lost",
        "Rebound (Q2 not asked)",
        "Other",
    ]
    cols = {
        "Stable": SEQ[1],
        "Rebound, perception intact": SERIES[1],
        "Rebound, perception lost": SERIES[7],
        "Rebound (Q2 not asked)": SERIES[3],
        "Other": MUTED,
    }
    t = tax.groupby(["dataset", "model", "outcome"]).size().unstack(fill_value=0)
    t = t.div(t.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(t) + 2.2))
    y = np.arange(len(t))[::-1]
    left = np.zeros(len(t))
    for o in order:
        if o in t:
            ax.barh(y, t[o], left=left, color=cols[o], label=o, edgecolor=SURF, linewidth=2, height=0.7)
            for yy, l, v in zip(y, left, t[o]):
                if v > 0.06:
                    ax.text(
                        l + v / 2,
                        yy,
                        f"{v:.0%}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=INK if o == "Stable" else "#ffffff",
                    )
            left += t[o].to_numpy()
    ax.set_yticks(y, [f"{m}  ({d})" for d, m in t.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of competence-set items under contra_2 (plain false caption)")
    ax.set_title("Fig 5. Outcome taxonomy per model")
    ax.grid(axis="y", visible=False)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, ncol=3, fontsize=8, loc="lower center")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIGDIR / "fig5_taxonomy.png")
    plt.close(fig)


def fig_forest(h1, colors):
    d = h1[h1.test == "contra_2"].copy()
    if not len(d):
        return
    d["label"] = d.model + " · " + d.dataset + "  vs " + d.ref
    d = d.sort_values(["ref", "dataset", "model"])
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(d) + 1.4))
    y = np.arange(len(d))[::-1]
    for yy, r in zip(y, d.itertuples()):
        ax.plot([r.OR_lo, r.OR_hi], [yy, yy], color=colors[r.model], lw=2)
        ax.plot(r.OR, yy, "o", color=colors[r.model], ms=8, mec=SURF, mew=2)
        ax.text(
            1.02,
            yy,
            f"OR {r.OR:.1f}  ΔRR {r.diff:+.2f}  p_holm {r.p_holm:.2g}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8,
            color=INK2,
        )
    ax.axvline(1, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(y, d.label, fontsize=8)
    ax.set_xlabel(
        "paired (McNemar) odds ratio of naming the distractor, cluster-bootstrap 95% CI, log scale"
    )
    ax.set_title("Fig 6. Plain false caption (contra_2) vs neutral / no text")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_forest.png", bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(q4, competent_items, items):
    d = q4[(q4.condition == "contra_2") & (q4.cap_tpl == 0) & (q4.q_tpl == 0)].merge(
        competent_items[["model", "item_id"]], on=["model", "item_id"]
    )
    d = d.merge(items[["item_id", "verb_key"]], on="item_id")
    d["verb"] = d.dataset.str[:5] + " · " + d.verb_key
    t = d.groupby(["verb", "model"]).rebind.agg(["mean", "count"]).reset_index()
    piv = t.pivot(index="verb", columns="model", values="mean")
    cnt = t.pivot(index="verb", columns="model", values="count")
    fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 4, 0.34 * piv.shape[0] + 1.6))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    im = ax.imshow(piv.to_numpy(float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v, n = piv.iat[i, j], cnt.iat[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}\n(n={int(n)})",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=INK if v < 0.55 else "#ffffff",
                )
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Reassignment Rate under contra_2")
    ax.set_title("Fig 7. Model x action heatmap")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig7_heatmap.png")
    plt.close(fig)


def write_rebinding_gallery(tax, items, n=20):
    """HTML gallery of rebinding examples (error analysis, 20 per model)."""
    it = items.set_index("item_id")
    parts = [
        "<!doctype html><meta charset=utf-8><title>Rebinding examples</title><style>body{font-family:sans-serif;background:#fcfcfb;color:#0b0b0b;margin:16px}"
        ".g{display:flex;flex-wrap:wrap;gap:12px}.c{width:300px;font-size:12px}.c img{width:300px;border:1px solid #e1e0d9}</style>"
    ]
    for m, g in tax[tax.rebind == 1].groupby("model"):
        parts.append(f"<h2>{html.escape(m)}</h2><div class=g>")
        for r in g.sample(min(n, len(g)), random_state=0).itertuples():
            row = it.loc[r.item_id]
            img = (ROOT / row.image_marked).resolve().as_uri()
            parts.append(
                f"<div class=c><img src='{img}'><br><b>{html.escape(r.item_id)}</b> · {html.escape(row.verb_key)}<br>"
                f"actor {row.actor_letter}, p_true={r.p_true:.2f}, Q2 ok={r.q2_ok}, Q3 flag={r.q3_flag}</div>"
            )
        parts.append("</div>")
    (REPORTS / "rebinding_gallery.html").write_text("\n".join(parts), encoding="utf-8")



def hypothesis_verdicts(h1, trend, gee, h34, probe):
    rows = []
    if len(h1):
        p = h1[h1.test == "contra_2"]
        sup = []
        for (m, ds), g in p.groupby(["model", "dataset"]):
            ok = (g.p_holm < 0.05).all() and (g["diff"] > 0).all() and len(g) == 2
            sup.append(ok)
        rows.append(
            (
                "H1 text rebinding",
                f"{sum(sup)}/{len(sup)} model×dataset cells support (Holm p<.05 vs neutral AND vs none, ΔRR>0)",
            )
        )
    if len(trend):
        pos = (trend.z > 0).mean()
        g_txt = "; ".join(
            f"{ds}: OR/level {g.get('OR_per_level', float('nan')):.2f} [{g.get('OR_lo', float('nan')):.2f}, {g.get('OR_hi', float('nan')):.2f}], p={g.get('p', float('nan')):.2g}"
            for ds, g in gee.items()
            if "OR_per_level" in g
        )
        rows.append(
            (
                "H2 dose-response",
                f"positive trend in {pos:.0%} of cells; monotone in {trend.monotone.mean():.0%}. GEE: {g_txt}",
            )
        )
    if len(h34) and "P_q2ok_rebind" in h34:
        s = h34.dropna(subset=["P_q2ok_rebind"])
        most = (s.P_q2ok_rebind_lo > 0.5).sum()
        dr = "; ".join(
            f"{r.model}/{r.dataset}: DR {format_number(r.DR, 2)} [{format_number(r.DR_lo, 2)}, {format_number(r.DR_hi, 2)}]"
            for r in s.itertuples()
            if hasattr(r, "DR")
        )
        rows.append(
            ("H3 dissociation", f"P(Q2 ok | rebind) lower bound > .5 in {most}/{len(s)} cells. {dr}")
        )
    if len(h34) and "OR_flag" in h34:
        s = h34.dropna(subset=["OR_flag"])
        noprot = ((s.OR_flag_lo <= 1) | (s.OR_flag > 0.5)).sum()
        rows.append(
            (
                "H4 awareness does not protect",
                f"no strong protection in {noprot}/{len(s)} cells (OR rebind|flag vs no flag: "
                + "; ".join(
                    f"{r.model}/{r.dataset} {r.OR_flag:.2f} [{r.OR_flag_lo:.2f}, {r.OR_flag_hi:.2f}]"
                    for r in s.itertuples()
                )
                + ")",
            )
        )
    rows.append(("H5 probe (stretch)", probe or "not run (src/probe_model.py)"))
    return pd.DataFrame(rows, columns=["hypothesis", "result"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma list (default: all non-mock results)")
    ap.add_argument("--audited-only", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)
    models = a.models.split(",") if a.models else None
    items = analysis.load_items()
    res = analysis.load_results(models)
    if a.audited_only:
        keep = set(items[items.audit_keep == True].item_id)  # noqa: E712
        res = res[res.item_id.isin(keep)]
    res = res[res.item_id.isin(items.item_id)]
    items = items[items.item_id.isin(res.item_id.unique()) | (items.dataset == "real")]
    colors = model_colors(res.model.unique())

    q4 = analysis.summarize_binding_answers(res)
    competent_items = analysis.competence_set(res, q4)
    rr = analysis.reassignment_rates(q4, competent_items)
    cap = analysis.capability(res, q4)
    bias = analysis.letter_bias(res)
    h1 = analysis.test_caption_effects(q4, competent_items, a.boot)
    trend, gee = analysis.test_confidence_trend(q4, competent_items, items)
    tax = analysis.taxonomy(res, q4, competent_items)
    h34 = analysis.summarize_perception_and_awareness(tax)
    rob = analysis.robustness(res, competent_items)
    diff = analysis.difficulty_regression(q4, competent_items, items)
    probe_p = REPORTS / "tables" / "probe_summary.json"
    probe = json.loads(probe_p.read_text()).get("verdict") if probe_p.exists() else None

    funnel = fig_funnel()
    fig_eda(items[items.item_id.isin(res.item_id.unique())])
    fig_capability(cap, colors)
    plot_dose_response(rr, colors)
    fig_taxonomy(tax)
    fig_forest(h1, colors)
    fig_heatmap(q4, competent_items, items)
    write_rebinding_gallery(tax, items)

    for name, t in [
        ("rr", rr),
        ("capability", cap),
        ("letter_bias", bias),
        ("h1", h1),
        ("h2_trend", trend),
        ("h3_h4", h34),
        ("taxonomy_items", tax),
        ("competence_set", competent_items),
    ] + [(f"robust_{k}", v) for k, v in rob.items()]:
        t.to_csv(TABDIR / f"{name}.csv", index=False)

    ver = hypothesis_verdicts(h1, trend, gee, h34, probe)
    S_sizes = competent_items.groupby("model").size().rename("|S_m|").reset_index()
    n_items = res.groupby(["model", "dataset"]).item_id.nunique().rename("items").reset_index()
    tax_share = tax.groupby(["model", "dataset", "outcome"]).size().unstack(fill_value=0).reset_index()
    real_n = int((items.dataset == "real").sum())
    md = [
        f"# Results{' (audited-clean real items)' if a.audited_only else ''}",
        "",
        "Generated by `src/analyze_results.py`. Definitions are in `PREREGISTRATION.md`.",
        "",
        f"Models: {', '.join(sorted(res.model.unique()))}",
        "",
        "## Hypotheses",
        markdown_table(ver),
        "## Sample",
        markdown_table(n_items.merge(S_sizes, on="model")),
        "Power: maximum 95% CI half-width on a proportion",
        markdown_table(analysis.power_note(tuple(sorted({100, 200, 400, 800, real_n} - {0})))),
        "## Funnel (real subset)",
        markdown_table(funnel) if funnel is not None else "_(not built)_\n",
        "![](figures/fig1_funnel.png)\n\n![](figures/fig2_eda.png)\n",
        "## Capability panel",
        markdown_table(cap[["model", "dataset", "task", "acc", "lo", "hi", "n"]]),
        "Letter/position bias under `none` (order averaging cancels it in the main metric):",
        markdown_table(bias),
        "![](figures/fig3_capability.png)\n",
        "## Reassignment rate by condition (on S_m)",
        markdown_table(
            rr[["model", "dataset", "condition", "n", "RR", "lo", "hi", "ERBS", "mean_logit_diff"]]
        ),
        "![](figures/fig4_dose_response.png)\n",
        "## H1: paired comparisons",
        markdown_table(h1.drop(columns=["b", "c"], errors="ignore")),
        "![](figures/fig6_forest.png)\n",
        "## H2: trend",
        markdown_table(trend),
    ]
    for ds, g in gee.items():
        md += [
            f"GEE ({ds}): `{g.get('formula')}`  ",
            f"OR per confidence level = {format_number(g.get('OR_per_level'))} "
            f"[{format_number(g.get('OR_lo'))}, {format_number(g.get('OR_hi'))}], p = {format_number(g.get('p'), 4)}, "
            f"n_obs = {g.get('n_obs')}, clusters = {g.get('n_clusters')} {g.get('error', '')}",
            "",
        ]
    md += [
        "## H3/H4 and taxonomy (contra_2)",
        markdown_table(h34),
        markdown_table(tax_share),
        "![](figures/fig5_taxonomy.png)\n",
        "## Robustness",
    ]
    for k, v in rob.items():
        md += [f"### {k}", markdown_table(v)]
    md += [
        "## Error analysis",
        "![](figures/fig7_heatmap.png)\n",
        "Difficulty features (GEE, odds ratio per SD, rebind under contra_2):",
        markdown_table(diff.reset_index().rename(columns={"index": "feature"}))
        if isinstance(diff, pd.DataFrame)
        else f"_{diff}_\n",
        "Examples: `reports/rebinding_gallery.html` (20 per model).",
        "",
        "## Extra analyses",
        "Bayesian Beta-Binomial reassignment rates and PCA of hidden states: `reports/results_extras.md` "
        "(`src/bayes_and_pca.py`, exploratory, not pre-registered).",
        "",
    ]
    if a.audited_only and any(
        pd.read_csv(p, dtype=str)
        .get("notes", pd.Series(dtype=str))
        .fillna("")
        .str.startswith("PLACEHOLDER")
        .any()
        for p in (REPORTS / "audit").glob("labels_*.csv")
    ):
        md.insert(
            1,
            "\n> **WARNING: some audit labels are simulated PLACEHOLDERS. "
            "This file is a pipeline check, not a result.**",
        )
    (REPORTS / ("results_audited.md" if a.audited_only else "results.md")).write_text(
        "\n".join(md), encoding="utf-8"
    )
    print(ver.to_string(index=False))
    print("wrote", REPORTS / "results.md")


if __name__ == "__main__":
    main()
