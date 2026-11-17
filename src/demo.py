"""Live demo: pick an image, watch a VLM's answer to "who is doing the action?" move as the caption changes.

    python src/demo.py                          # live, SmolVLM2 on the GPU (~35 s load, ~2 s per image)
    python src/demo.py --model qwen2.5-vl-3b    # live, Qwen-3B 4-bit
    python src/demo.py --replay                 # no GPU: replays the recorded full-run answers from results/
    then open http://localhost:8800

For the chosen item it asks Q4 ("Who is <action>, A or B?") under every caption condition, in both
option orders, and plots P(correct person) averaged over the two orders, exactly as in the experiment.
It also asks Q2 (is the action present?) and Q3 (does the caption match?) under contra_4. In live mode
the audience can type their own caption. Stdlib HTTP server only; the GPU is used by one request at a time.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, REPORTS, RESULTS, ROOT, build_queries, load_config, make_caption, resize_max  # noqa: E402

CONDS = ["none", "neutral", "consistent", "contra_1", "contra_2", "contra_3", "contra_4"]
LABEL = {"none": "no caption", "neutral": "neutral", "consistent": "true caption", "contra_1": "false, hedged",
         "contra_2": "false, plain", "contra_3": "false, \"definitely\"", "contra_4": "false, \"verified\""}


class Demo:
    def __init__(self, a):
        self.cfg = load_config()
        self.model_name = a.model
        self.replay = a.replay
        fr = [pd.read_parquet(p) for p in (PROC / "real" / "items_real.parquet", PROC / "synthetic" / "items_synthetic.parquet") if p.exists()]
        self.items = pd.concat(fr, ignore_index=True).set_index("item_id", drop=False)
        self.lock = threading.Lock()
        self.rec = None
        rp = RESULTS / f"{a.model}.parquet"
        if rp.exists():
            r = pd.read_parquet(rp)
            self.rec = r[(r.block == "main") & (r.error.isna())]
        if self.replay and self.rec is None:
            raise SystemExit(f"--replay needs {rp}")
        # suggested items: ones this model rebinds under contra_2 in the full run, plus a few it resists
        self.suggest = []
        tp = REPORTS / "tables" / "taxonomy_items.csv"
        if tp.exists():
            t = pd.read_csv(tp)
            t = t[(t.model == a.model)]
            for ds in ("real", "synthetic"):
                g = t[t.dataset == ds]
                self.suggest += g[g.rebind == 1].item_id.head(8).tolist() + g[g.rebind == 0].item_id.head(3).tolist()
        if not self.suggest:
            self.suggest = self.items.groupby("dataset").head(8).item_id.tolist()
        self.B = None
        if not self.replay:
            from backends import HFLocal
            spec = self.cfg["models"][a.model]
            print(f"loading {spec['hf_id']} ...", flush=True)
            self.B = HFLocal(spec["hf_id"], spec.get("dtype", "float16"), spec.get("quant", "none"),
                             self.cfg["runtime"]["qwen_max_pixels"], image_processor=spec.get("image_processor"))
            self.max_side = spec.get("max_image_side", self.cfg["runtime"]["max_image_side"])
            # warm-up so the first click on stage is fast
            it = self.items.iloc[0].to_dict()
            self._ask(it, build_queries(self.cfg, it, "full", {"main"})[0])
            print("model ready", flush=True)

    # ------------------------------------------------------------------ model calls
    def _img(self, it):
        return resize_max(Image.open(ROOT / it["image_marked"]).convert("RGB"), self.max_side)

    def _ask(self, it, q, prompt=None):
        return self.B.ask(self._img(it), prompt or q.prompt, q.options, q.system)

    def run(self, item_id, custom=None):
        it = self.items.loc[item_id].to_dict()
        qs = build_queries(self.cfg, it, "full", {"main"})
        out = {"item_id": item_id, "dataset": it["dataset"], "action": f"{it['ing']} {it['obj_np']}",
               "actor": it["actor_letter"], "mode": "replay" if self.replay else "live", "model": self.model_name, "rows": []}
        if self.replay:
            d = self.rec[self.rec.item_id == item_id]
            if d.empty:
                out["error"] = "this item is not in the recorded run"
                return out
            for c in CONDS:
                g = d[(d.question == "q4") & (d.condition == c)]
                if len(g):
                    out["rows"].append({"cond": c, "label": LABEL[c], "caption": make_caption(self.cfg, c, it, "marker"),
                                        "p": float(g.p_correct.mean())})
            for qn, key in (("q2", "q2"), ("q3", "q3")):
                g = d[(d.question == qn) & (d.condition.isin(["contra_4", "contra_2"]))].sort_values("condition", ascending=False)
                if len(g):
                    po = g.iloc[0].p_options
                    po = json.loads(po) if isinstance(po, str) else po
                    out[key] = {"cond": g.iloc[0].condition, "p_yes": float(po["Yes"]) if po else None, "answer": g.iloc[0].answer}
            return out
        with self.lock:
            for c in CONDS:
                ps = [self._ask(it, q)["p_options"][q.correct] for q in qs if q.question == "q4" and q.condition == c]
                out["rows"].append({"cond": c, "label": LABEL[c], "caption": make_caption(self.cfg, c, it, "marker"), "p": sum(ps) / len(ps)})
            if custom:
                wrap = self.cfg["prompts"]["caption_wrapper"].format(caption=custom.strip())
                ps = []
                for q in qs:
                    if q.question == "q4" and q.condition == "none":
                        ps.append(self._ask(it, q, wrap + "\n" + q.prompt)["p_options"][q.correct])
                out["rows"].append({"cond": "custom", "label": "your caption", "caption": custom.strip(), "p": sum(ps) / len(ps)})
            for qn in ("q2", "q3"):
                q = next(q for q in qs if q.question == qn and q.condition == "contra_4")
                r = self._ask(it, q)
                out[qn] = {"cond": "contra_4", "p_yes": r["p_options"]["Yes"], "answer": r["answer"]}
        return out


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Caption vs Image</title><style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mut:#898781;--line:#e1e0d9;--ok:#2a78d6;--bad:#e34948;--chip:#f0efec}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--ink:#f0efec;--ink2:#c3c2b7;--mut:#898781;--line:#33322e;--ok:#5a9cf0;--bad:#f06a69;--chip:#22211e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
header h1{font-size:19px;margin:0}header span{color:var(--ink2);font-size:13px}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);gap:22px;padding:18px 20px;max-width:1400px}
@media (max-width:900px){main{grid-template-columns:1fr}}
#img{width:100%;border-radius:8px;border:1px solid var(--line);background:var(--chip);min-height:200px}
.truth{font-size:17px;margin:10px 0 4px}.truth b{color:var(--ok)}
.thumbs{display:flex;gap:6px;overflow-x:auto;padding:8px 0}.thumbs img{height:64px;border-radius:4px;cursor:pointer;border:2px solid transparent}
.thumbs img.sel{border-color:var(--ok)}
.row{display:grid;grid-template-columns:150px 1fr 56px;gap:10px;align-items:center;margin:7px 0}
.row .lab{font-size:13px;color:var(--ink2)}.row .cap{grid-column:1/-1;font-size:12px;color:var(--mut);margin:-4px 0 2px 160px}
.track{position:relative;height:26px;background:var(--chip);border-radius:4px;overflow:hidden}
.bar{position:absolute;inset:0 auto 0 0;border-radius:4px;transition:width .5s ease}
.mid{position:absolute;left:50%;top:0;bottom:0;border-left:1px dashed var(--mut)}
.val{font-variant-numeric:tabular-nums;text-align:right;font-weight:600}
.ok .bar{background:var(--ok)}.bad .bar{background:var(--bad)}
.badges{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}.badge{background:var(--chip);border-radius:6px;padding:8px 10px;font-size:13px}
form{display:flex;gap:8px;margin-top:14px}input[type=text]{flex:1;font:inherit;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
button{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid var(--line);background:var(--ink);color:var(--bg);cursor:pointer}
button[disabled]{opacity:.5}select{font:inherit;padding:6px;border-radius:6px;background:var(--bg);color:var(--ink);border:1px solid var(--line)}
.legend{font-size:12px;color:var(--ink2)}#status{color:var(--ink2);font-size:13px;min-height:1.4em}
</style></head><body>
<header><h1>Does the caption override the image?</h1><span id=meta></span></header>
<main><section>
<div><select id=pick></select> <button id=rand type=button>Random item</button></div>
<div class=thumbs id=thumbs></div>
<img id=img alt="selected image with persons marked A and B">
<div class=truth id=truth></div><div id=status></div>
</section><section>
<div class=legend>Bar = model's probability for the <b>correct</b> person, averaged over both answer orders. Left of the dashed line = the model names the wrong person.</div>
<div id=rows></div><div class=badges id=badges></div>
<form id=f><input type=text id=custom placeholder="Type your own caption, e.g. 'Person B is clearly kicking the ball.'"><button id=go>Run my caption</button></form>
</section></main>
<script>
const $=id=>document.getElementById(id);let INFO=null,cur=null;
async function j(u,o){const r=await fetch(u,o);return r.json()}
function draw(d){
  $("truth").innerHTML=`Ground truth: <b>${d.actor}</b> is ${d.action}. <span class=legend>(${d.dataset}, ${d.item_id})</span>`;
  if(d.error){$("rows").innerHTML=`<p>${d.error}</p>`;return}
  $("rows").innerHTML=d.rows.map(r=>`<div class="row ${r.p>=.5?'ok':'bad'}"><div class=lab>${r.label}</div>
   <div class=track><div class=bar style="width:${(r.p*100).toFixed(1)}%"></div><div class=mid></div></div><div class=val>${(r.p*100).toFixed(0)}%</div>
   <div class=cap>${r.caption?('"'+r.caption.replace(/</g,'&lt;')+'"'):'(image only)'}</div></div>`).join("");
  const b=[];
  if(d.q2)b.push(`<div class=badge>Q2 with ${d.q2.cond} caption: "Is anyone ${d.action.split(' ')[0]} ...?" → <b>${d.q2.answer}</b>${d.q2.p_yes!=null?` (P(yes)=${(d.q2.p_yes*100).toFixed(0)}%)`:''}<br><span class=legend>Yes = it still sees the action</span></div>`);
  if(d.q3)b.push(`<div class=badge>Q3 with ${d.q3.cond} caption: "Does the caption match?" → <b>${d.q3.answer}</b>${d.q3.p_yes!=null?` (P(yes)=${(d.q3.p_yes*100).toFixed(0)}%)`:''}<br><span class=legend>Yes = it did not notice the lie</span></div>`);
  $("badges").innerHTML=b.join("");
}
async function run(id,custom){
  cur=id;$("img").src="/img?id="+encodeURIComponent(id);$("pick").value=id;
  document.querySelectorAll(".thumbs img").forEach(t=>t.classList.toggle("sel",t.dataset.id===id));
  $("status").textContent=INFO.mode==="live"?"Running the model ...":"Loading recorded answers ...";$("go").disabled=true;
  const t0=performance.now();
  const d=await j("/api/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({item_id:id,custom:custom||null})});
  $("status").textContent=`${d.mode==="live"?"Live":"Recorded"} answers from ${d.model} in ${((performance.now()-t0)/1000).toFixed(1)} s`;
  $("go").disabled=INFO.mode!=="live";draw(d);
}
(async()=>{INFO=await j("/api/info");
  $("meta").textContent=`model: ${INFO.model} · mode: ${INFO.mode}`;
  $("pick").innerHTML=INFO.items.map(i=>`<option value="${i.id}">${i.dataset}: ${i.action} (${i.id})</option>`).join("");
  $("thumbs").innerHTML=INFO.suggest.map(id=>`<img loading=lazy data-id="${id}" src="/img?id=${encodeURIComponent(id)}" alt="${id}">`).join("");
  $("thumbs").onclick=e=>{if(e.target.dataset.id)run(e.target.dataset.id)};
  $("pick").onchange=e=>run(e.target.value);
  $("rand").onclick=()=>run(INFO.items[Math.floor(Math.random()*INFO.items.length)].id);
  $("f").onsubmit=e=>{e.preventDefault();if(cur&&$("custom").value.trim())run(cur,$("custom").value)};
  if(INFO.mode!=="live"){$("custom").disabled=true;$("custom").placeholder="Custom captions need live mode (run without --replay)";$("go").disabled=true}
  run(INFO.suggest[0]);
})();
</script></body></html>"""


def make_handler(demo: Demo):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/api/info":
                its = demo.items
                if demo.rec is not None:
                    its = its[its.item_id.isin(set(demo.rec.item_id))]
                info = {"model": demo.model_name, "mode": "replay" if demo.replay else "live", "suggest": demo.suggest,
                        "items": [{"id": r.item_id, "dataset": r.dataset, "action": f"{r.ing} {r.obj_np}"} for r in its.itertuples()]}
                return self._send(200, json.dumps(info).encode(), "application/json")
            if u.path == "/img":
                iid = parse_qs(u.query).get("id", [""])[0]
                if iid not in demo.items.index:
                    return self._send(404, b"no such item", "text/plain")
                im = Image.open(ROOT / demo.items.loc[iid, "image_marked"]).convert("RGB")
                b = io.BytesIO()
                im.save(b, "JPEG", quality=88)
                return self._send(200, b.getvalue(), "image/jpeg")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/api/run":
                return self._send(404, b"not found", "text/plain")
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            iid = body.get("item_id")
            if iid not in demo.items.index:
                return self._send(400, b'{"error":"unknown item"}', "application/json")
            custom = (body.get("custom") or "")[:300] or None
            try:
                out = demo.run(iid, custom)
            except Exception as e:  # noqa: BLE001
                it = demo.items.loc[iid]
                out = {"item_id": iid, "dataset": it.dataset, "actor": it.actor_letter, "action": f"{it.ing} {it.obj_np}",
                       "mode": "replay" if demo.replay else "live", "model": demo.model_name, "error": f"{type(e).__name__}: {e}"}
            self._send(200, json.dumps(out).encode(), "application/json")
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvlm2-2.2b")
    ap.add_argument("--replay", action="store_true", help="no GPU: show recorded answers from results/<model>.parquet")
    ap.add_argument("--port", type=int, default=8800)
    a = ap.parse_args()
    demo = Demo(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(demo))
    print(f"demo on http://localhost:{a.port}  ({'replay' if a.replay else 'live'}, {a.model})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
