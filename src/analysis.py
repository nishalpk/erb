"""Statistics for the ERB study (imported by 05_analysis.py and 05_analysis.ipynb)."""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import PROC, RESULTS

CONTRA = ["contra_1", "contra_2", "contra_3", "contra_4"]
COND_ORDER = ["none", "neutral", "consistent"] + CONTRA
LEVEL = {"none": 0, "neutral": 0, "consistent": 0, "contra_1": 1, "contra_2": 2, "contra_3": 3, "contra_4": 4}


# ------------------------------------------------------------------ basic stats
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def mcnemar(b, c):
    """Exact McNemar (binomial on discordant pairs). b: ref=0,test=1 ; c: ref=1,test=0."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def cochran_armitage(levels, k, n):
    """Two-sided Cochran-Armitage trend test. levels: scores, k: events, n: totals."""
    levels, k, n = map(lambda x: np.asarray(x, float), (levels, k, n))
    N, K = n.sum(), k.sum()
    if N == 0 or K in (0, N):
        return np.nan, 1.0
    p = K / N
    T = np.sum(levels * (k - n * p))
    var = p * (1 - p) * (np.sum(n * levels ** 2) - np.sum(n * levels) ** 2 / N)
    if var <= 0:
        return np.nan, 1.0
    z = T / math.sqrt(var)
    return z, 2 * stats.norm.sf(abs(z))


def odds_ratio_2x2(a, b, c, d):
    """a: exposed&event, b: exposed&no-event, c: unexposed&event, d: unexposed&no-event (Haldane 0.5)."""
    a, b, c, d = (x + 0.5 for x in (a, b, c, d))
    orr = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return orr, math.exp(math.log(orr) - 1.96 * se), math.exp(math.log(orr) + 1.96 * se)


# ------------------------------------------------------------------ loading
def load_items():
    fr = []
    aud, plain = PROC / "real" / "items_real_audited.parquet", PROC / "real" / "items_real.parquet"
    for p in [aud if aud.exists() else plain, PROC / "synthetic" / "items_synthetic.parquet"]:
        if p.exists():
            fr.append(pd.read_parquet(p))
    items = pd.concat(fr, ignore_index=True)
    if "audit_keep" not in items:
        items["audit_keep"] = None
    items["audit_keep"] = items["audit_keep"].astype(object)
    items.loc[items["dataset"] == "synthetic", "audit_keep"] = True
    return items


def load_results(models=None, include_mock=False):
    fr = []
    for p in sorted(RESULTS.glob("*.parquet")):
        name = p.stem
        if "_smoke" in name and not (models and name in models):
            continue
        if models and name not in models:
            continue
        if name.startswith("mock") and not include_mock and not (models and name in models):
            continue
        df = pd.read_parquet(p)
        df = df[df["error"].isna()].copy()
        df["model"] = name
        fr.append(df)
    if not fr:
        raise SystemExit("no results/*.parquet found")
    res = pd.concat(fr, ignore_index=True)
    res["p_options"] = res["p_options"].apply(lambda s: json.loads(s) if isinstance(s, str) else s)
    return res


# ------------------------------------------------------------------ per-item tables
def q4_table(res, block="main", scheme="marker"):
    """Order-averaged Q4: one row per (model, item, condition, cap_tpl, q_tpl)."""
    d = res[(res.question == "q4") & (res.block == block) & (res.scheme == scheme)].copy()
    d["correct_ans"] = d["answer"] == d["correct"]
    d["names_distractor"] = d["answer"] == d["distractor"]
    g = d.groupby(["model", "item_id", "dataset", "condition", "cap_tpl", "q_tpl"], as_index=False).agg(
        p_true=("p_correct", "mean"), n_orders=("order", "nunique"), both_correct=("correct_ans", "all"),
        any_distractor=("names_distractor", "any"), logit_diff=("logit_diff", "mean"),
        readout=("readout", "first"))
    g["rebind"] = (g["p_true"] < 0.5).astype(int)
    g["tie"] = np.isclose(g["p_true"], 0.5)
    return g


def yesno_table(res, question):
    d = res[(res.question == question) & (res.block == "main")].copy()
    d["ok"] = (d["answer"] == d["correct"]).astype(int)
    return d[["model", "item_id", "dataset", "condition", "ok", "p_correct"]]


def competence(res, q4):
    """S_m: correct on `none` Q4 in both orders and on Q2 under `none` (if asked)."""
    base = q4[(q4.condition == "none") & (q4.cap_tpl == 0) & (q4.q_tpl == 0)]
    s = base[(base.both_correct) & (base.n_orders == 2)][["model", "item_id"]].copy()
    q2 = yesno_table(res, "q2")
    q2n = q2[q2.condition == "none"]
    out = []
    for m, sm in s.groupby("model"):
        q2m = q2n[q2n.model == m]
        if len(q2m):
            ok = set(q2m[q2m.ok == 1].item_id)
            sm = sm[sm.item_id.isin(ok)]
            sm = sm.assign(s_def="Q4 none (both orders) + Q2 none")
        else:
            sm = sm.assign(s_def="Q4 none (both orders) only")
        out.append(sm)
    return pd.concat(out, ignore_index=True) if out else s.assign(s_def="")


def rr_table(q4, S, by=("model", "dataset")):
    d = q4[(q4.cap_tpl == 0) & (q4.q_tpl == 0)].merge(S[["model", "item_id"]], on=["model", "item_id"])
    rows = []
    for key, g in d.groupby(list(by) + ["condition"]):
        k, n = int(g.rebind.sum()), len(g)
        p, lo, hi = wilson(k, n)
        rows.append({**dict(zip(list(by) + ["condition"], key)), "n": n, "k": k, "RR": p, "lo": lo, "hi": hi,
                     "ERBS": 1 - p, "mean_logit_diff": g.logit_diff.mean(), "ties": int(g.tie.sum())})
    out = pd.DataFrame(rows)
    out["condition"] = pd.Categorical(out["condition"], COND_ORDER, ordered=True)
    return out.sort_values(list(by) + ["condition"])


# ------------------------------------------------------------------ H1
def paired_effect(d, ref, test, n_boot=2000, seed=0):
    """McNemar + paired RR difference + conditional OR (b/c) with item-cluster bootstrap CIs."""
    w = d.pivot_table(index="item_id", columns="condition", values="rebind", aggfunc="first")
    if ref not in w or test not in w:
        return None
    w = w[[ref, test]].dropna()
    if len(w) == 0:
        return None
    x, y = w[ref].to_numpy(int), w[test].to_numpy(int)
    b, c = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    xb, yb = x[idx], y[idx]
    diff_b = yb.mean(1) - xb.mean(1)
    bb = ((xb == 0) & (yb == 1)).sum(1) + 0.5
    cb = ((xb == 1) & (yb == 0)).sum(1) + 0.5
    or_b = bb / cb
    return {"n": len(x), "RR_ref": x.mean(), "RR_test": y.mean(), "diff": y.mean() - x.mean(),
            "diff_lo": np.percentile(diff_b, 2.5), "diff_hi": np.percentile(diff_b, 97.5),
            "b": b, "c": c, "OR": (b + 0.5) / (c + 0.5),
            "OR_lo": np.percentile(or_b, 2.5), "OR_hi": np.percentile(or_b, 97.5), "p": mcnemar(b, c)}


def h1(q4, S, n_boot=2000):
    d = q4[(q4.cap_tpl == 0) & (q4.q_tpl == 0)].merge(S[["model", "item_id"]], on=["model", "item_id"])
    rows = []
    for (m, ds), g in d.groupby(["model", "dataset"]):
        for ref in ("neutral", "none"):
            for test in CONTRA:
                r = paired_effect(g, ref, test, n_boot)
                if r:
                    rows.append({"model": m, "dataset": ds, "ref": ref, "test": test, **r})
    out = pd.DataFrame(rows)
    if len(out):
        prim = out.test == "contra_2"
        out["p_holm"] = np.nan
        out.loc[prim, "p_holm"] = holm(out.loc[prim, "p"])
    return out


# ------------------------------------------------------------------ H2
def h2(q4, S, items):
    d = q4[(q4.cap_tpl == 0) & (q4.q_tpl == 0) & q4.condition.isin(CONTRA)].merge(S[["model", "item_id"]], on=["model", "item_id"])
    d["level"] = d.condition.map(LEVEL)
    trend = []
    for (m, ds), g in d.groupby(["model", "dataset"]):
        t = g.groupby("level").rebind.agg(["sum", "count"]).reindex([1, 2, 3, 4]).dropna()
        z, p = cochran_armitage(t.index, t["sum"], t["count"])
        rr = (t["sum"] / t["count"]).round(3).tolist()
        mono = all(np.diff(rr) >= -1e-9)
        trend.append({"model": m, "dataset": ds, "z": z, "p": p, "RR_by_level": rr, "monotone": mono})
    trend = pd.DataFrame(trend)
    gee = {}
    feats = items[["item_id", "verb_key", "min_person_area", "actor_side", "person_dist"]]
    dd = d.merge(feats, on="item_id", how="left")
    for ds, g in dd.groupby("dataset"):
        gee[ds] = fit_gee(g)
    return trend, gee


def fit_gee(g):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    g = g.copy()
    g["area_z"] = (g.min_person_area - g.min_person_area.mean()) / (g.min_person_area.std() or 1)
    parts = ["level"]
    for c in ("model", "verb_key", "actor_side"):
        if g[c].nunique() > 1:
            parts.append(f"C({c})")
    parts.append("area_z")
    formula = "rebind ~ " + " + ".join(parts)
    if g.rebind.nunique() < 2:
        return {"formula": formula, "error": "no variation in outcome"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mod = smf.gee(formula, groups="item_id", data=g, family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
        except Exception as e:  # noqa: BLE001
            return {"formula": formula, "error": str(e)[:200]}
    b, se = mod.params["level"], mod.bse["level"]
    return {"formula": formula, "beta_level": b, "OR_per_level": math.exp(b),
            "OR_lo": math.exp(b - 1.96 * se), "OR_hi": math.exp(b + 1.96 * se), "p": mod.pvalues["level"],
            "n_obs": int(mod.nobs), "n_clusters": int(g.item_id.nunique()), "summary": str(mod.summary())}


# ------------------------------------------------------------------ H3/H4 + taxonomy
def taxonomy(res, q4, S, cond="contra_2"):
    d = q4[(q4.condition == cond) & (q4.cap_tpl == 0) & (q4.q_tpl == 0)].merge(S[["model", "item_id"]], on=["model", "item_id"])
    q2 = yesno_table(res, "q2")
    q3 = yesno_table(res, "q3")
    d = d.merge(q2[q2.condition == cond][["model", "item_id", "ok"]].rename(columns={"ok": "q2_ok"}), on=["model", "item_id"], how="left")
    d = d.merge(q3[q3.condition == cond][["model", "item_id", "ok"]].rename(columns={"ok": "q3_flag"}), on=["model", "item_id"], how="left")

    def lab(r):
        if r.tie:
            return "Other"
        if not r.rebind:
            return "Stable"
        if pd.isna(r.q2_ok):
            return "Rebound (Q2 not asked)"
        return "Rebound, perception intact" if r.q2_ok == 1 else "Rebound, perception lost"
    d["outcome"] = d.apply(lab, axis=1)
    return d


def h3_h4(tax):
    rows = []
    for (m, ds), g in tax.groupby(["model", "dataset"]):
        g = g[~g.tie]
        rb, st = g[g.rebind == 1], g[g.rebind == 0]
        r = {"model": m, "dataset": ds, "n": len(g), "n_rebind": len(rb)}
        if g.q2_ok.notna().any() and g.q3_flag.notna().any():
            k = int(((rb.q2_ok == 1) & (rb.q3_flag == 1)).sum())
            r["DR"], r["DR_lo"], r["DR_hi"] = wilson(k, len(rb))
        if g.q2_ok.notna().any():
            r["P_q2ok_rebind"], r["P_q2ok_rebind_lo"], r["P_q2ok_rebind_hi"] = wilson(int((rb.q2_ok == 1).sum()), len(rb))
            r["P_q2ok_stable"], _, _ = wilson(int((st.q2_ok == 1).sum()), len(st))
            tab = [[int((rb.q2_ok == 1).sum()), int((rb.q2_ok == 0).sum())], [int((st.q2_ok == 1).sum()), int((st.q2_ok == 0).sum())]]
            r["p_q2_fisher"] = stats.fisher_exact(tab)[1] if min(map(sum, tab)) > 0 else np.nan
        if g.q3_flag.notna().any():
            fl, nf = g[g.q3_flag == 1], g[g.q3_flag == 0]
            r["P_rebind_flag"] = fl.rebind.mean() if len(fl) else np.nan
            r["P_rebind_noflag"] = nf.rebind.mean() if len(nf) else np.nan
            a, b = int(fl.rebind.sum()), int((1 - fl.rebind).sum())
            c, d = int(nf.rebind.sum()), int((1 - nf.rebind).sum())
            r["OR_flag"], r["OR_flag_lo"], r["OR_flag_hi"] = odds_ratio_2x2(a, b, c, d)
            r["p_flag_fisher"] = stats.fisher_exact([[a, b], [c, d]])[1]
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ capability panel
def capability(res, q4):
    rows = []
    base = q4[(q4.condition == "none") & (q4.cap_tpl == 0) & (q4.q_tpl == 0)]
    for (m, ds), g in base.groupby(["model", "dataset"]):
        p, lo, hi = wilson(int(g.both_correct.sum()), len(g))
        rows.append({"model": m, "dataset": ds, "task": "Q4 binding, no text (both orders)", "acc": p, "lo": lo, "hi": hi, "n": len(g)})
    for q, cond, label in [("q1", "none", "Q1 entity identity"), ("q2", "none", "Q2 action present"),
                           ("q2_foil", "none", "Q2 foil (absent action)"), ("q3", "consistent", "Q3 true caption = match"),
                           ("q3", "contra_2", "Q3 false caption = conflict")]:
        d = res[(res.question == q) & (res.block == "main") & (res.condition == cond)].copy()
        if q == "q1":
            d = d.groupby(["model", "dataset", "item_id"], as_index=False).apply(
                lambda x: pd.Series({"ok": int((x.answer == x.correct).all())}), include_groups=False)
        else:
            d["ok"] = (d.answer == d.correct).astype(int)
        for (m, ds), g in d.groupby(["model", "dataset"]):
            p, lo, hi = wilson(int(g.ok.sum()), len(g))
            rows.append({"model": m, "dataset": ds, "task": label, "acc": p, "lo": lo, "hi": hi, "n": len(g)})
    return pd.DataFrame(rows)


def letter_bias(res):
    d = res[(res.question == "q4") & (res.block == "main") & (res.condition == "none") & (res.scheme == "marker")]
    out = d.groupby(["model", "dataset"]).apply(lambda g: pd.Series({
        "P(answer=A)": (g.answer == "A").mean(), "P(answer=first option)": (g.answer == g.options.apply(lambda s: json.loads(s)[0] if isinstance(s, str) else s[0])).mean(),
        "mean_option_mass": g.option_mass.mean()}), include_groups=False)
    return out.reset_index()


# ------------------------------------------------------------------ robustness
def robustness(res, S):
    out = {}
    q4r = q4_table(res, block="robust")
    if len(q4r):
        d = q4r.merge(S[["model", "item_id"]], on=["model", "item_id"])
        t = d[d.condition == "contra_2"].groupby(["model", "dataset", "cap_tpl", "q_tpl"]).rebind.mean().reset_index()
        out["paraphrase"] = t
        out["paraphrase_var"] = t.groupby(["model", "dataset"]).rebind.agg(["mean", "std", "min", "max"]).reset_index()
    q4a = q4_table(res, block="ablation")
    q4m = q4_table(res, block="main")
    if len(q4a):
        rows = []
        for (m, ds), g in q4a.merge(S[["model", "item_id"]], on=["model", "item_id"]).groupby(["model", "dataset"]):
            base = q4m[(q4m.model == m) & (q4m.condition == "contra_2") & q4m.item_id.isin(g.item_id)]
            both = pd.concat([base.assign(condition="contra_2"), g.assign(condition="contra_2+instr")])
            r = paired_effect(both, "contra_2", "contra_2+instr")
            if r:
                rows.append({"model": m, "dataset": ds, **r})
        out["ablation"] = pd.DataFrame(rows)
    q4l = q4_table(res, block="lr", scheme="lr")
    if len(q4l):
        Sl = q4l[(q4l.condition == "none") & q4l.both_correct][["model", "item_id"]]
        d = q4l.merge(Sl, on=["model", "item_id"])
        lr = d.groupby(["model", "dataset", "condition"]).rebind.agg(["mean", "count"]).reset_index().assign(scheme="left/right")
        mk = q4m[q4m.item_id.isin(q4l.item_id.unique())].merge(S[["model", "item_id"]], on=["model", "item_id"])
        mk = mk[mk.condition.isin(lr.condition.unique())].groupby(["model", "dataset", "condition"]).rebind.agg(["mean", "count"]).reset_index().assign(scheme="A/B markers")
        out["scheme"] = pd.concat([mk, lr], ignore_index=True)
    return out


# ------------------------------------------------------------------ error analysis
def difficulty_regression(q4, S, items, cond="contra_2"):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = q4[(q4.condition == cond) & (q4.cap_tpl == 0) & (q4.q_tpl == 0)].merge(S[["model", "item_id"]], on=["model", "item_id"])
    d = d.merge(items[["item_id", "min_person_area", "person_dist", "obj_area", "obj_actor_dist", "log_area_ratio"]], on="item_id")
    if d.rebind.nunique() < 2 or len(d) < 30:
        return None
    for c in ["min_person_area", "person_dist", "obj_area", "obj_actor_dist", "log_area_ratio"]:
        d[c + "_z"] = (d[c] - d[c].mean()) / (d[c].std() or 1)
    f = "rebind ~ min_person_area_z + person_dist_z + obj_area_z + obj_actor_dist_z + log_area_ratio_z"
    if d.model.nunique() > 1:
        f += " + C(model)"
    if d.dataset.nunique() > 1:
        f += " + C(dataset)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = smf.gee(f, groups="item_id", data=d, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        except Exception as e:  # noqa: BLE001
            return str(e)
    t = pd.DataFrame({"OR": np.exp(m.params), "lo": np.exp(m.conf_int()[0]), "hi": np.exp(m.conf_int()[1]), "p": m.pvalues})
    return t[t.index.str.endswith("_z")]


def power_note(ns=(100, 200, 400, 800)):
    return pd.DataFrame({"n": ns, "max 95% CI half-width (pp)": [round(100 * 1.96 * math.sqrt(0.25 / n), 1) for n in ns]})
