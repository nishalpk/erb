"""D2: manual audit of every real candidate (CVC5), agreement and filter precision.

    python src/02_audit_sample.py prepare --annotators R1,R2,R3 --shared 100
        -> reports/audit/audit_<R>.html  (open in a browser, label, click "Export CSV")
    python src/02_audit_sample.py score
        -> reads reports/audit/labels_<R>.csv, writes data/processed/real/items_real_audited.parquet
           and reports/audit/audit_summary.md (Fleiss' kappa on the shared subset, precision + Wilson CI)
    python src/02_audit_sample.py triage
        -> reports/audit/triage.csv: items EVERY model gets wrong with no text -> back to manual review
           (never auto-deleted: that would bias the benchmark toward easy items)

Keep rule: exactly one person does the action = yes, it is the annotated actor, the other person could
NOT plausibly be doing it = no, a second such object = no. Any "unclear" -> drop.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, REPORTS, RESULTS, ROOT, load_config  # noqa: E402

AUD = REPORTS / "audit"
ITEMS = PROC / "real" / "items_real.parquet"
FIELDS = ["exactly_one", "who", "other_could", "second_object"]


def fleiss_kappa(counts: np.ndarray) -> float:
    """counts: (n_items, n_categories) number of raters choosing each category."""
    counts = np.asarray(counts, float)
    n = counts.sum(1)[0]
    p_j = counts.sum(0) / counts.sum()
    P_i = (np.sum(counts ** 2, 1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), np.sum(p_j ** 2)
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else float("nan")


def audit_image(row) -> str:
    """Marked image + the target object box (cyan, dashed look) as base64 JPEG."""
    im = Image.open(ROOT / row["image_marked"]).convert("RGB")
    d = ImageDraw.Draw(im)
    b = [float(v) for v in row["obj_box"]]
    for k in range(0, 3):
        d.rectangle([b[0] - k, b[1] - k, b[2] + k, b[3] + k], outline=(0, 220, 255))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ERB audit {ann}</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--acc:#2a78d6}}
body{{font-family:system-ui,sans-serif;background:var(--bg);color:var(--ink);margin:0 16px 80px;max-width:980px}}
.card{{border:1px solid var(--line);border-radius:8px;padding:12px;margin:14px 0;display:flex;gap:16px;flex-wrap:wrap}}
.card img{{max-width:520px;width:100%;border-radius:4px}} .q{{margin:6px 0}} .q b{{display:block;font-size:13px}}
label{{margin-right:10px;font-size:13px}} .done{{border-color:var(--acc)}} header{{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line)}}
button{{font-size:14px;padding:6px 12px}} small{{color:var(--ink2)}}
</style></head><body><header><b>Annotator {ann}</b> · <span id=prog></span> labelled ·
<button onclick="exportCsv()">Export CSV</button> <small>Saved in this browser as you go. Cyan box = the target object.</small></header>
<p><small>For each image: is exactly one of A/B doing the action with the cyan-boxed object? Who? Could the <i>other</i> person plausibly be doing it too,
or is there a second such object, even partly hidden? Answer "unclear" whenever unsure; unclear items are dropped.</small></p>
<div id=list></div><script>
const ITEMS={items};const ANN="{ann}";const KEY="erb_audit_"+ANN;
let S={{}};try{{S=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{}}
const Q=[["exactly_one","Exactly one of A/B is <i>ACTION</i>?",["yes","no","unclear"]],["who","Who?",["A","B","unclear"]],
["other_could","Could the other person plausibly be doing it too?",["yes","no","unclear"]],["second_object","Is there a second such object?",["yes","no","unclear"]]];
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(S))}}catch(e){{}};prog()}}
function prog(){{let n=ITEMS.filter(it=>S[it.item_id]&&Q.every(q=>S[it.item_id][q[0]])).length;document.getElementById("prog").textContent=n+"/"+ITEMS.length;
ITEMS.forEach(it=>document.getElementById("c_"+it.item_id).classList.toggle("done",!!(S[it.item_id]&&Q.every(q=>S[it.item_id][q[0]]))))}}
const L=document.getElementById("list");
ITEMS.forEach(it=>{{const c=document.createElement("div");c.className="card";c.id="c_"+it.item_id;
let h=`<img src="data:image/jpeg;base64,${{it.img}}"><div><b>${{it.item_id}}</b><br><small>action: ${{it.action}}</small>`;
Q.forEach(([k,t,opts])=>{{h+=`<div class=q><b>${{t.replace("ACTION",it.action)}}</b>`+opts.map(o=>`<label><input type=radio data-id="${{it.item_id}}" data-k="${{k}}" name="${{it.item_id}}__${{k}}" value="${{o}}" ${{(S[it.item_id]||{{}})[k]===o?"checked":""}}> ${{o}}</label>`).join("")+"</div>"}});
h+=`<div class=q><b>Notes</b><input style="width:280px" data-id="${{it.item_id}}" data-k="notes" value="${{((S[it.item_id]||{{}}).notes||"").replace(/"/g,"&quot;")}}"></div></div>`;
c.innerHTML=h;L.appendChild(c)}});
function upd(e){{const id=e.target.dataset.id,k=e.target.dataset.k;if(!id||!k)return;S[id]=S[id]||{{}};S[id][k]=e.target.value;save()}}
L.addEventListener("change",upd);L.addEventListener("input",e=>{{if(e.target.dataset.k==="notes")upd(e)}});
function exportCsv(){{const cols=["item_id",...Q.map(q=>q[0]),"notes"];const esc=v=>'"'+String(v??"").replace(/"/g,'""')+'"';
const rows=[cols.join(",")].concat(ITEMS.map(it=>cols.map(c=>c==="item_id"?it.item_id:esc((S[it.item_id]||{{}})[c])).join(",")));
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\\n")],{{type:"text/csv"}}));a.download="labels_"+ANN+".csv";a.click()}}
prog();</script></body></html>"""


def prepare(a, cfg):
    items = pd.read_parquet(ITEMS)
    rng = random.Random(cfg["seed"] + 3)
    anns = a.annotators.split(",")
    ids = list(items.item_id)
    rng.shuffle(ids)
    # shared subset stratified by action
    by_verb = items.set_index("item_id").verb_key
    strata: dict[str, list] = {}
    for i in ids:
        strata.setdefault(by_verb[i], []).append(i)
    shared, k = [], 0
    while len(shared) < min(a.shared, len(ids)):
        for v in sorted(strata):
            if k < len(strata[v]) and len(shared) < a.shared:
                shared.append(strata[v][k])
        k += 1
    rest = [i for i in ids if i not in set(shared)]
    assign = {r: list(shared) for r in anns}
    for j, i in enumerate(rest):
        assign[anns[j % len(anns)]].append(i)
    AUD.mkdir(parents=True, exist_ok=True)
    rows = items.set_index("item_id")
    plan = []
    for r, lst in assign.items():
        rng.shuffle(lst)
        payload = [{"item_id": i, "action": f"{rows.loc[i, 'ing']} {rows.loc[i, 'obj_np']}", "img": audit_image(rows.loc[i])} for i in lst]
        (AUD / f"audit_{r}.html").write_text(HTML.format(ann=r, items=json.dumps(payload)), encoding="utf-8")
        plan += [{"annotator": r, "item_id": i, "shared": i in set(shared)} for i in lst]
    pd.DataFrame(plan).to_csv(AUD / "assignment.csv", index=False)
    print(f"{len(items)} items; shared {len(shared)}; per annotator: " + ", ".join(f"{r}={len(v)}" for r, v in assign.items()))
    print("open:", *[str(AUD / f"audit_{r}.html") for r in anns], sep="\n  ")


def keep_rule(row, actor_letter):
    return (row["exactly_one"] == "yes" and row["who"] == actor_letter and row["other_could"] == "no"
            and row["second_object"] == "no")


def score(a, cfg):
    from analysis import wilson
    items = pd.read_parquet(ITEMS)
    actor = items.set_index("item_id").actor_letter
    labs = []
    for p in sorted(AUD.glob("labels_*.csv")):
        d = pd.read_csv(p, dtype=str).fillna("")
        d["annotator"] = p.stem.split("_", 1)[1]
        labs.append(d)
    if not labs:
        raise SystemExit("no reports/audit/labels_*.csv yet")
    L = pd.concat(labs, ignore_index=True)
    L = L[L[FIELDS].ne("").all(axis=1)]
    L["keep"] = [keep_rule(dict(zip(FIELDS, [getattr(r, f) for f in FIELDS])), actor[r.item_id]) for r in L.itertuples()]
    assign = pd.read_csv(AUD / "assignment.csv")
    shared = set(assign[assign.shared].item_id)
    n_ann = assign.annotator.nunique()

    lines = ["# Audit summary", ""]
    S = L[L.item_id.isin(shared)]
    full = S.groupby("item_id").annotator.nunique()
    S = S[S.item_id.isin(full[full == n_ann].index)]
    if len(S):
        for f in FIELDS + ["keep"]:
            cats = sorted(S[f].astype(str).unique())
            cnt = S.groupby("item_id")[f].apply(lambda s: [list(s.astype(str)).count(c) for c in cats])
            k = fleiss_kappa(np.array(cnt.tolist()))
            lines.append(f"- Fleiss' kappa ({f}), {len(cnt)} shared items x {n_ann} raters: **{k:.3f}**")
    # item decision: shared items need every rater to keep; single-labelled items use the one label
    dec = L.groupby("item_id").keep.all()
    k, n = int(dec.sum()), len(dec)
    p, lo, hi = wilson(k, n)
    lines += ["", f"- Items reviewed: {n} of {len(items)}", f"- Automatic-filter precision (kept / reviewed): **{p:.3f}** "
              f"(95% Wilson CI {lo:.3f}-{hi:.3f})", ""]
    reasons = L[~L.keep].assign(reason=lambda d: np.select(
        [d.exactly_one != "yes", d.who != d.item_id.map(actor), d.other_could != "no", d.second_object != "no"],
        ["not exactly one", "wrong actor", "other could do it", "second object"], "other"))
    lines += ["Drop reasons (label rows):", "", "\n".join(f"- {k}: {v}" for k, v in reasons.reason.value_counts().items()) if len(reasons) else "none", ""]
    (AUD / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    out = items.copy()
    out["audit_keep"] = out.item_id.map(dec).astype(object)
    out["audit_status"] = np.where(out.audit_keep.isna(), "pending", np.where(out.audit_keep == True, "clean", "dropped"))  # noqa: E712
    out.to_parquet(PROC / "real" / "items_real_audited.parquet", index=False)
    print("\n".join(lines))
    # funnel last row
    fp = PROC / "real" / "funnel.csv"
    if fp.exists():
        f = pd.read_csv(fp)
        f = f[~f.stage.str.startswith("CVC5")]
        row = {"stage": "CVC5: manual audit kept", "total": k}
        for c in f.columns:
            if c not in row:
                row[c] = int(out[(out.audit_keep == True) & (out.source == c)].shape[0]) if "source" in out else 0  # noqa: E712
        pd.concat([f, pd.DataFrame([row])], ignore_index=True).to_csv(fp, index=False)


def triage(a, cfg):
    rows = []
    for p in RESULTS.glob("*.parquet"):
        if p.stem.startswith("mock"):
            continue
        d = pd.read_parquet(p)
        d = d[(d.question == "q4") & (d.block == "main") & (d.condition == "none") & (d.dataset == "real") & d.error.isna()]
        g = d.groupby("item_id").p_correct.mean()
        rows.append((g < 0.5).rename(p.stem))
    if not rows:
        raise SystemExit("no model results yet")
    W = pd.concat(rows, axis=1)
    all_wrong = W[W.all(1)].index
    pd.DataFrame({"item_id": all_wrong, "n_models": W.shape[1]}).to_csv(AUD / "triage.csv", index=False)
    print(f"{len(all_wrong)} items wrong for all {W.shape[1]} models (no text) -> reports/audit/triage.csv for re-review")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "score", "triage"])
    ap.add_argument("--annotators", default="R1,R2,R3")
    ap.add_argument("--shared", type=int, default=100)
    a = ap.parse_args()
    cfg = load_config()
    AUD.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "score": score, "triage": triage}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
