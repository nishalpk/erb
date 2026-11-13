"""D7 (stretch): mechanistic check on one model (default Qwen2.5-VL-3B, 4-bit on a 6 GB GPU).

1. Linear probe (H5): last-token residual stream at every layer under `none` and `contra_3`.
   Logistic probe for "true actor is A" trained on `none`, tested on `contra_3` of held-out items
   (GroupKFold by item). If mid-layer accuracy stays high on items whose OUTPUT flips, the information
   is still there and is rebound later.
2. Activation patching: clean = `consistent` ("A is kicking..."), corrupt = `contra_2` ("B is kicking...").
   Both prompts have identical length and differ only at the entity token, so positions align exactly.
   The image comes first, so image-token states are identical in both runs (patching them is a no-op by
   construction); we patch at {entity token, rest of caption, question, final token} x every layer and
   measure recovery of logit_diff = logit(true) - logit(distractor).
3. Attention knockout: on the corrupt run, block attention from the final token to the entity token in a
   sliding window of layers and measure the change in logit_diff.

    python src/06_probe_patch.py --model qwen2.5-vl-3b --n 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from backends import HFLocal  # noqa: E402
from common import FIG, PROC, REPORTS, ROOT, build_queries, load_config, resize_max  # noqa: E402


def find_layers(model):
    best = None
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "ModuleList" and len(mod) > 4 and ("language" in name or "model.layers" in name or name.endswith("layers")):
            if "visual" in name or "vision" in name:
                continue
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise RuntimeError("could not locate decoder layers")
    return best[1]


def q4_prompt(cfg, item, cond):
    qs = [q for q in build_queries(cfg, item, "full", {"main"}) if q.question == "q4" and q.condition == cond and q.order == 0]
    return qs[0]


def load_items(n, cfg):
    fr = []
    for p in (PROC / "synthetic" / "items_synthetic.parquet", PROC / "real" / "items_real.parquet"):
        if p.exists():
            fr.append(pd.read_parquet(p))
    items = pd.concat(fr, ignore_index=True).sample(frac=1, random_state=cfg["seed"])
    return items.groupby("dataset", group_keys=False).head(n).to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-vl-3b")
    ap.add_argument("--n", type=int, default=200, help="items per dataset")
    ap.add_argument("--patch-n", type=int, default=60, help="items used for patching / knockout")
    ap.add_argument("--window", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config()
    spec = cfg["models"][a.model]
    B = HFLocal(spec["hf_id"], spec.get("dtype", "float16"), spec.get("quant", "none"), cfg["runtime"]["qwen_max_pixels"],
               image_processor=spec.get("image_processor"))
    torch = B.torch
    layers = find_layers(B.model)
    nL = len(layers)
    items = load_items(a.n, cfg)
    max_side = spec.get("max_image_side", cfg["runtime"]["max_image_side"])

    def run(item, cond, hooks=None, hidden=False):
        q = q4_prompt(cfg, item, cond)
        img = resize_max(Image.open(ROOT / q.image_path).convert("RGB"), max_side)
        inp = B.build_inputs(img, q.prompt, q.system)
        handles = [layers[l].register_forward_hook(h) for l, h in (hooks or [])]
        try:
            out = B.forward_logits(inp, output_hidden_states=hidden)
        finally:
            for h in handles:
                h.remove()
        lg = out.logits[0, -1].float()
        sc = {o: torch.logsumexp(lg[B.option_ids(o)], 0).item() for o in q.options}
        return q, inp, out, sc[q.correct] - sc[q.distractor]

    # ---------------------------------------------------------------- 1. probe
    feats = {"none": [], "contra_3": []}
    meta = []
    for it in tqdm(items, desc="probe: caching residuals"):
        row = {"item_id": it["item_id"], "dataset": it["dataset"], "y": int(it["actor_letter"] == "A")}
        for cond in ("none", "contra_3"):
            q, inp, out, ld = run(it, cond, hidden=True)
            hs = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states]).numpy()
            feats[cond].append(hs)
            row[f"ld_{cond}"] = ld
        meta.append(row)
    meta = pd.DataFrame(meta)
    meta["flipped"] = (meta.ld_none > 0) & (meta.ld_contra_3 < 0)
    Xn, Xc = np.stack(feats["none"]), np.stack(feats["contra_3"])
    y = meta.y.to_numpy()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    gkf = GroupKFold(n_splits=5)
    rows = []
    for l in range(Xn.shape[1]):
        pred_c = np.zeros(len(y))
        pred_n = np.zeros(len(y))
        for tr, te in gkf.split(Xn[:, l], y, groups=meta.item_id):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            clf.fit(Xn[tr, l], y[tr])
            pred_c[te] = clf.predict(Xc[te, l])
            pred_n[te] = clf.predict(Xn[te, l])
        ok_c = pred_c == y
        rows.append({"layer": l, "acc_none": (pred_n == y).mean(), "acc_contra3": ok_c.mean(),
                     "acc_contra3_flipped": ok_c[meta.flipped].mean() if meta.flipped.any() else np.nan,
                     "n_flipped": int(meta.flipped.sum())})
    probe = pd.DataFrame(rows)

    # ---------------------------------------------------------------- 2. patching
    sel = [it for it in items][: a.patch_n]
    groups = ["entity", "caption_rest", "question", "final"]
    rec = np.full((len(sel), nL, len(groups)), np.nan)
    knock = np.full((len(sel), nL), np.nan)
    ld_rows = []
    for k, it in enumerate(tqdm(sel, desc="patching")):
        qc, inp_c, out_c, ld_clean = run(it, "consistent", hidden=True)
        qx, inp_x, _, ld_corr = run(it, "contra_2")
        ids_c, ids_x = inp_c["input_ids"][0], inp_x["input_ids"][0]
        if ids_c.shape != ids_x.shape:
            continue
        diff = (ids_c != ids_x).nonzero().flatten().tolist()
        if not diff or ld_clean - ld_corr < 0.5:
            ld_rows.append({"item_id": it["item_id"], "ld_clean": ld_clean, "ld_corrupt": ld_corr, "used": False})
            continue
        ld_rows.append({"item_id": it["item_id"], "ld_clean": ld_clean, "ld_corrupt": ld_corr, "used": True})
        T = ids_c.shape[0]
        e0, e1 = diff[0], diff[-1] + 1
        text_tail = B.tok.decode(ids_x[e1:].tolist())
        # caption ends at the first newline after the entity
        nl = [i for i in range(e1, T) if "\n" in B.tok.decode([int(ids_x[i])])]
        cap_end = nl[0] if nl else e1
        pos = {"entity": list(range(e0, e1)), "caption_rest": list(range(e1, cap_end + 1)),
               "question": list(range(cap_end + 1, T - 1)), "final": [T - 1]}
        clean_h = [h[0].detach() for h in out_c.hidden_states[1:]]  # output of each decoder layer
        for l in range(nL):
            for gi, gname in enumerate(groups):
                P = pos[gname]
                if not P:
                    continue

                def hook(mod, args, output, l=l, P=P):
                    hs = output[0] if isinstance(output, tuple) else output
                    hs = hs.clone()
                    hs[0, P] = clean_h[l][P].to(hs.dtype)
                    return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
                _, _, _, ld_p = run(it, "contra_2", hooks=[(l, hook)])
                rec[k, l, gi] = (ld_p - ld_corr) / (ld_clean - ld_corr)
        # ---------------------------------------------------------------- 3. knockout
        for l0 in range(0, nL - a.window + 1):
            def pre(mod, args, kwargs, E=list(range(e0, e1)), T=T):
                m = kwargs.get("attention_mask")
                dt = B.model.dtype if B.model.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16
                neg = torch.finfo(dt).min
                if m is None or m.dim() != 4:
                    m = torch.full((1, 1, T, T), neg, dtype=dt, device=B.model.device).triu(1)
                else:
                    m = m.clone()
                    if m.dtype == torch.bool:
                        m = torch.where(m, torch.zeros((), dtype=dt, device=m.device), torch.tensor(neg, dtype=dt, device=m.device))
                m[..., T - 1, E] = neg
                kwargs["attention_mask"] = m
                return args, kwargs
            hs = []
            try:
                for l in range(l0, l0 + a.window):
                    attn = getattr(layers[l], "self_attn", None)
                    hs.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
                _, _, _, ld_k = run(it, "contra_2")
                knock[k, l0] = ld_k - ld_corr
            except Exception as ex:  # noqa: BLE001
                knock[k, l0] = np.nan
                if k == 0 and l0 == 0:
                    print("knockout unsupported for this model:", ex)
            finally:
                for h in hs:
                    h.remove()

    # ---------------------------------------------------------------- outputs
    tab = REPORTS / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    probe.to_csv(tab / "probe_by_layer.csv", index=False)
    meta.to_csv(tab / "probe_items.csv", index=False)
    pd.DataFrame(ld_rows).to_csv(tab / "patch_items.csv", index=False)
    rec_mean = np.nanmean(rec, 0)
    pd.DataFrame(rec_mean, columns=groups).rename_axis("layer").to_csv(tab / "patch_recovery.csv")
    pd.DataFrame({"window_start": range(nL), "delta_logit_diff": np.nanmean(knock, 0)}).to_csv(tab / "knockout.csv", index=False)

    mid = probe.iloc[len(probe) // 4: 3 * len(probe) // 4]
    best = mid.loc[mid.acc_contra3_flipped.idxmax()] if mid.acc_contra3_flipped.notna().any() else None
    from analysis import wilson
    if best is not None and best.n_flipped > 0:
        k_ok = round(best.acc_contra3_flipped * best.n_flipped)
        p, lo, hi = wilson(int(k_ok), int(best.n_flipped))
        verdict = (f"{'SUPPORTED' if lo > 0.5 else 'NOT supported'}: best mid layer {int(best.layer)} decodes the true actor on "
                   f"{p:.2f} [{lo:.2f}, {hi:.2f}] of {int(best.n_flipped)} output-flipped items ({a.model}; probe trained on `none`)")
    else:
        verdict = f"inconclusive: no output flips under contra_3 among {len(meta)} items ({a.model})"
    (tab / "probe_summary.json").write_text(json.dumps({"model": a.model, "verdict": verdict, "n_items": len(meta),
                                                        "n_flipped": int(meta.flipped.sum()), "n_layers": nL}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axs[0]
    ax.plot(probe.layer, probe.acc_none, "-", color="#2a78d6", lw=2, label="test on none")
    ax.plot(probe.layer, probe.acc_contra3, "-", color="#eb6834", lw=2, label="test on contra_3 (all)")
    ax.plot(probe.layer, probe.acc_contra3_flipped, "-", color="#e34948", lw=2, label=f"contra_3, output flipped (n={int(meta.flipped.sum())})")
    ax.axhline(0.5, color="#898781", ls="--", lw=1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("layer")
    ax.set_ylabel("probe accuracy (true actor)")
    ax.set_title("Linear probe trained on `none`")
    ax.legend(fontsize=8, frameon=False)
    ax = axs[1]
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])
    im = ax.imshow(rec_mean.T, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    ax.set_yticks(range(len(groups)), groups)
    ax.set_xlabel("layer")
    ax.set_title("Patching clean→corrupt: logit-diff recovery")
    fig.colorbar(im, ax=ax)
    ax = axs[2]
    ax.plot(range(nL), np.nanmean(knock, 0), "-o", color="#1baf7a", ms=3)
    ax.axhline(0, color="#898781", lw=1)
    ax.set_xlabel(f"first layer of {a.window}-layer window")
    ax.set_ylabel("Δ logit_diff (positive = toward true actor)")
    ax.set_title("Knockout: final token ↛ entity token")
    fig.suptitle(f"Fig 8. Mechanistic check ({a.model})", fontweight="bold")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig8_probe_patch.png")
    print(verdict)


if __name__ == "__main__":
    main()
