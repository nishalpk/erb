"""Serve the interactive benchmark demo in live or replay mode."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from common import PROC, REPORTS, RESULTS, ROOT, build_queries, load_config, make_caption, load_model_image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


LOCAL_MODELS = ["qwen2.5-vl-3b", "smolvlm2-2.2b", "llava-ov-0.5b"]
CONDS = ["none", "neutral", "consistent", "contra_1", "contra_2", "contra_3", "contra_4"]
TAB = REPORTS / "tables"
SRC_CACHE = PROC / "real" / "source_labels.json"
PROBE_LAYER = 25
LIVE_LIE = "contra_2"  # live probe: the plain lie (the probe itself is trained on no-caption states only)



def build_source_cache(items: pd.DataFrame) -> dict:
    """COCO boxes + V-COCO action labels for every person/object in each real V-COCO source photo (one-off, ~1 min)."""
    want = {}
    for _, it in items[items.source == "vcoco"].iterrows():
        f = it.source_file
        want[int(f.rsplit("_", 1)[1].split(".")[0])] = it.item_id
    out = {}
    actions: dict[int, list] = {}
    for f in ("trainval", "test"):
        for act in json.loads((ROOT / "data" / "raw" / "vcoco" / f"vcoco_{f}.json").read_text()):
            for im, aid, lab in zip(act["image_id"], act["ann_id"], act["label"]):
                if im in want and lab:
                    actions.setdefault(aid, set()).add(act["action_name"])
    for split in ("val", "train"):
        print(f"reading COCO {split} annotations ...", flush=True)
        coco = json.loads(
            (ROOT / "data" / "raw" / "coco" / "annotations" / f"instances_{split}2014.json").read_text()
        )
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        for a in coco["annotations"]:
            if a["image_id"] in want:
                iid = want[a["image_id"]]
                out.setdefault(iid, []).append(
                    {
                        "ann_id": a["id"],
                        "category": cats[a["category_id"]],
                        "bbox": [round(v, 1) for v in a["bbox"]],
                        "actions": sorted(actions.get(a["id"], [])),
                    }
                )
        del coco
        gc.collect()
    SRC_CACHE.write_text(json.dumps(out))
    return out



class App:
    def __init__(self, a):
        self.cfg = load_config()
        self.replay = a.replay
        fr = [
            pd.read_parquet(PROC / "real" / "items_real.parquet"),
            pd.read_parquet(PROC / "synthetic" / "items_synthetic.parquet"),
        ]
        self.items = pd.concat(fr, ignore_index=True).set_index("item_id", drop=False)
        self.src = (
            json.loads(SRC_CACHE.read_text()) if SRC_CACHE.exists() else build_source_cache(self.items)
        )
        self.funnel = pd.read_csv(PROC / "real" / "funnel.csv")
        self.tax = pd.read_csv(TAB / "taxonomy_items.csv")
        self.rr = pd.read_csv(TAB / "rr.csv")
        self.h1 = pd.read_csv(TAB / "h1.csv")
        self.h34 = pd.read_csv(TAB / "h3_h4.csv")
        self.bayes = pd.read_csv(TAB / "bayes_rr.csv")
        self.probe_tab = pd.read_csv(TAB / "probe_by_layer.csv")
        self.patch = pd.read_csv(TAB / "patch_recovery.csv")
        self.probe_summary = json.loads((TAB / "probe_summary.json").read_text())
        self.gee = {}
        txt = (REPORTS / "results.md").read_text(encoding="utf-8")
        for ds in ("real", "synthetic"):
            m = re.search(
                rf"GEE \({ds}\).*?\nOR per confidence level = ([\d.]+) \[([\d.]+), ([\d.]+)\].*?n_obs = (\d+), clusters = (\d+)",
                txt,
            )
            if m:
                self.gee[ds] = {
                    "OR": float(m[1]),
                    "lo": float(m[2]),
                    "hi": float(m[3]),
                    "n_obs": int(m[4]),
                    "clusters": int(m[5]),
                }
        self.rec_cache: dict[str, pd.DataFrame] = {}
        self.lock = threading.Lock()
        self.backend = None
        self.model = None
        self.status = "idle"
        self.npz = None
        self.load(a.model)

    def load(self, model):
        if model not in LOCAL_MODELS:
            raise ValueError(model)
        if self.replay:
            self.model = model
            return
        with self.lock:
            if self.model == model and self.backend is not None:
                return
            self.status = f"loading {model}"
            if self.backend is not None:
                import torch

                del self.backend
                self.backend = None
                gc.collect()
                torch.cuda.empty_cache()
            from backends import HFLocal

            spec = self.cfg["models"][model]
            print(f"loading {spec['hf_id']} ...", flush=True)
            t0 = time.time()
            self.backend = HFLocal(
                spec["hf_id"],
                spec.get("dtype", "float16"),
                spec.get("quant", "none"),
                self.cfg["runtime"]["qwen_max_pixels"],
                image_processor=spec.get("image_processor"),
            )
            self.max_side = spec.get("max_image_side", self.cfg["runtime"]["max_image_side"])
            self.model = model
            self.status = "ready"
            print(f"{model} ready in {time.time() - t0:.0f} s", flush=True)

    def recorded_answers(self, model):
        if model not in self.rec_cache:
            r = pd.read_parquet(RESULTS / f"{model}.parquet")
            self.rec_cache[model] = r[(r.block == "main") & (r.error.isna())].set_index("qid")
        return self.rec_cache[model]

    def curated_items(self, model):
        t = self.tax[self.tax.model == model]
        out = []
        for ds in ("real", "synthetic"):
            g = t[t.dataset == ds]
            reb = g[g.outcome == "Rebound, perception intact"].sort_values("p_true")
            out += [(i, "fooled") for i in reb.item_id.head(10 if ds == "real" else 6)]
            out += [
                (i, "resists")
                for i in g[g.outcome == "Stable"].sort_values("p_true", ascending=False).item_id.head(3)
            ]
        if model == "qwen2.5-vl-3b" and "vcoco_000000085007" in self.items.index:
            out = [("vcoco_000000085007", "fooled")] + [o for o in out if o[0] != "vcoco_000000085007"]
        res = []
        for iid, tag in out:
            it = self.items.loc[iid]
            res.append({"id": iid, "label": f"{it.dataset}: {it.ing} {it.obj_np}", "tag": tag})
        return res

    def item_info(self, iid):
        it = self.items.loc[iid].to_dict()
        qs = build_queries(self.cfg, it, "full", {"main"})
        checks = []
        if it["dataset"] == "real":
            checks = [
                [
                    "CVC1",
                    "exactly two people, clearly separable",
                    f"overlap IoU = {it['person_iou']:.2f}",
                    it["person_iou"] < 0.3,
                ],
                [
                    "CVC1",
                    "both people large enough to see",
                    f"smaller person = {it['min_person_area']:.1%} of image",
                    it["min_person_area"] >= 0.02,
                ],
                ["CVC1c", "detector finds no extra unlabelled person", "RT-DETR check", True],
                [
                    "CVC2",
                    "exactly one target object",
                    f"one {it['object']}; detector confirms: {bool(it['obj_detected'])}",
                    True,
                ],
                ["CVC3", "the action belongs to one person only", f"actor = {it['actor_letter']}", True],
                [
                    "CVC4",
                    "the other person has no conflicting label",
                    f"other person's labels: {it['distractor_labels'] or 'none'}",
                    True,
                ],
            ]
        src = None
        raw = None
        if it["dataset"] == "real":
            raw = f"data/raw/real_images/{it['source_file']}"
            if it["source"] == "vcoco":
                src = self.src.get(iid)
            else:
                src = [
                    {
                        "ann_id": "actor",
                        "category": "person",
                        "bbox": _xyxy_to_xywh(it["actor_box"]),
                        "actions": [it["verb"]],
                    },
                    {
                        "ann_id": "other",
                        "category": "person",
                        "bbox": _xyxy_to_xywh(it["distractor_box"]),
                        "actions": [x for x in str(it["distractor_labels"]).split(";") if x],
                    },
                    {
                        "ann_id": "object",
                        "category": it["object"],
                        "bbox": _xyxy_to_xywh(it["obj_box"]),
                        "actions": [],
                    },
                ]
        return {
            "id": iid,
            "dataset": it["dataset"],
            "source": it["source"],
            "source_file": it["source_file"],
            "action": f"{it['ing']} {it['obj_np']}",
            "ing": it["ing"],
            "obj": it["obj_np"],
            "verb_key": it["verb_key"],
            "verb": it["verb"],
            "actor": it["actor_letter"],
            "actor_side": it["actor_side"],
            "w": int(it["img_w"]),
            "h": int(it["img_h"]),
            "actor_box": list(map(float, it["actor_box"])),
            "distractor_box": list(map(float, it["distractor_box"])),
            "obj_box": list(map(float, it["obj_box"])),
            "raw": raw,
            "marked": it["image_marked"],
            "plain": it["image_plain"],
            "src": src,
            "checks": checks,
            "colors": [it.get("actor_color"), it.get("distractor_color")]
            if it["dataset"] == "synthetic"
            else None,
            "captions": [{"cond": c, "text": make_caption(self.cfg, c, it, "marker")} for c in CONDS],
            "queries": [
                {
                    "qid": q.qid,
                    "question": q.question,
                    "condition": q.condition,
                    "order": q.order,
                    "options": q.options,
                    "correct": q.correct,
                    "prompt": q.prompt,
                }
                for q in qs
            ],
        }

    def _load_marked_image(self, it):
        return load_model_image(ROOT / it["image_marked"], self.max_side)

    def ask(self, iid, qid=None, custom=None, order=0):
        it = self.items.loc[iid].to_dict()
        qs = build_queries(self.cfg, it, "full", {"main"})
        if custom is not None:
            q = next(q for q in qs if q.question == "q4" and q.condition == "none" and q.order == order)
            prompt = self.cfg["prompts"]["caption_wrapper"].format(caption=custom.strip()) + "\n" + q.prompt
        else:
            q = next(q for q in qs if q.qid == qid)
            prompt = q.prompt
        if self.replay:
            if custom is not None:
                return {"error": "Custom captions need live mode (start without --replay)."}
            rec = self.recorded_answers(self.model)
            if q.qid not in rec.index:
                return {"error": "not in the recorded run"}
            r = rec.loc[q.qid]
            po = json.loads(r.p_options) if isinstance(r.p_options, str) else r.p_options
            lg = json.loads(r.logits) if isinstance(r.logits, str) else r.logits
            return {
                "qid": q.qid,
                "answer": r.answer,
                "p": po,
                "logscore": lg,
                "top": None,
                "sec": float(r.latency_s or 0),
                "correct": q.correct,
                "prompt": prompt,
                "source": "recorded run",
            }
        import torch

        backend = self.backend
        with self.lock:
            t0 = time.time()
            inp = backend.build_inputs(self._load_marked_image(it), prompt, q.system)
            out = backend.forward_logits(inp)
            lg = out.logits[0, -1].float()
            lp = torch.log_softmax(lg, -1)
            sc = {o: torch.logsumexp(lp[backend.option_ids(o)], 0).item() for o in q.options}
            top = torch.topk(lp.exp(), 10)
            sec = time.time() - t0
            ntok = int(inp["input_ids"].shape[1])
            chat = backend.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
                add_generation_prompt=True,
            )
        m = max(sc.values())
        z = sum(math.exp(v - m) for v in sc.values())
        p = {o: math.exp(v - m) / z for o, v in sc.items()}
        return {
            "qid": q.qid,
            "answer": max(p, key=p.get),
            "p": p,
            "logscore": sc,
            "sec": sec,
            "n_tokens": ntok,
            "top": [
                {"tok": backend.tok.decode([int(i)]), "p": float(v)}
                for i, v in zip(top.indices, top.values)
            ],
            "option_ids": {o: backend.option_ids(o) for o in q.options},
            "correct": q.correct,
            "prompt": prompt,
            "chat": chat,
            "source": "live",
        }

    def dataset(self, model):
        def for_model(frame):
            return frame[frame.model == model]

        rr = for_model(self.rr)
        h1 = for_model(self.h1)
        h1 = h1[h1.test.isin(["contra_2", "contra_4"])]
        bay = for_model(self.bayes)
        allm = self.rr[self.rr.model.isin(LOCAL_MODELS) & self.rr.condition.isin(["contra_2", "contra_4"])]
        return {
            "model": model,
            "rr": rr[["dataset", "condition", "n", "k", "RR", "lo", "hi"]].to_dict("records"),
            "h1": h1[
                [
                    "dataset",
                    "ref",
                    "test",
                    "n",
                    "RR_ref",
                    "RR_test",
                    "diff",
                    "diff_lo",
                    "diff_hi",
                    "b",
                    "c",
                    "p",
                    "p_holm",
                ]
            ].to_dict("records"),
            "h34": for_model(self.h34).to_dict("records"),
            "bayes": bay[
                ["dataset", "condition", "n", "k", "alpha", "beta", "post_mean", "cri_lo", "cri_hi"]
            ].to_dict("records"),
            "gee": self.gee,
            "compare": allm[["model", "dataset", "condition", "RR", "lo", "hi", "n"]].to_dict("records"),
        }

    def probe_static(self):
        return {
            "layers": self.probe_tab.to_dict("records"),
            "patch": self.patch.to_dict("records"),
            "summary": self.probe_summary,
            "layer": PROBE_LAYER,
        }

    def probe_live(self, iid):
        if self.replay or self.model != "qwen2.5-vl-3b":
            return {
                "error": "The live probe needs Qwen2.5-VL-3B in live mode (the probe was trained on Qwen's hidden states). "
                "Switch the model to Qwen at the top of the page."
            }
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if self.npz is None:
            z = np.load(RESULTS / "probe_resid_qwen2.5-vl-3b.npz")
            self.npz = {"Xn": z["Xn"].astype(np.float32), "y": z["y"], "ids": z["item_id"]}
        it = self.items.loc[iid].to_dict()
        qs = build_queries(self.cfg, it, "full", {"main"})
        H, ans = {}, {}
        with self.lock:
            for cond in ("none", LIVE_LIE):
                q = next(q for q in qs if q.question == "q4" and q.condition == cond and q.order == 0)
                inp = self.backend.build_inputs(self._load_marked_image(it), q.prompt, q.system)
                o = self.backend.forward_logits(inp, output_hidden_states=True)
                H[cond] = torch.stack([h[0, -1].float().cpu() for h in o.hidden_states]).numpy()
                lg = torch.log_softmax(o.logits[0, -1].float(), -1)
                sc = {x: torch.logsumexp(lg[self.backend.option_ids(x)], 0).item() for x in q.options}
                ans[cond] = 1 / (1 + math.exp(sc[q.distractor] - sc[q.correct]))
        Xn, y, ids = self.npz["Xn"], self.npz["y"], self.npz["ids"]
        keep = ids != iid
        truth_is_A = it["actor_letter"] == "A"
        rows = []
        t0 = time.time()
        for l in range(Xn.shape[1]):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(
                Xn[keep, l], y[keep]
            )
            pa = clf.predict_proba(np.stack([H["none"][l], H[LIVE_LIE][l]]))[:, 1]
            pt = pa if truth_is_A else 1 - pa
            rows.append({"layer": l, "none": float(pt[0]), "lie": float(pt[1])})
        cos = (H["none"] * H[LIVE_LIE]).sum(1) / (
            np.linalg.norm(H["none"], axis=1) * np.linalg.norm(H[LIVE_LIE], axis=1)
        )
        return {
            "layers": rows,
            "cos": cos.tolist(),
            "answer_p_true": {"none": ans["none"], "lie": ans[LIVE_LIE]},
            "train_n": int(keep.sum()),
            "sec": time.time() - t0,
            "lie_caption": make_caption(self.cfg, LIVE_LIE, it, "marker"),
            "actor": it["actor_letter"],
        }


def _xyxy_to_xywh(b):
    b = list(map(float, b))
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


def _json_ready(value):
    """Convert NumPy values and non-finite floats for browser JSON."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)



def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(_json_ready(body), allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self._send(
                        200,
                        (Path(__file__).parent / "demo_app.html").read_bytes(),
                        "text/html; charset=utf-8",
                    )
                if u.path == "/api/state":
                    return self._send(
                        200,
                        {
                            "model": app.model,
                            "mode": "replay" if app.replay else "live",
                            "status": app.status,
                            "models": LOCAL_MODELS,
                            "curated": app.curated_items(app.model),
                            "all": [
                                {"id": i, "label": f"{r.dataset}: {r.ing} {r.obj_np}"}
                                for i, r in app.items.iterrows()
                            ],
                            "funnel": app.funnel.to_dict("records"),
                        },
                    )
                if u.path == "/api/item":
                    return self._send(200, app.item_info(q["id"]))
                if u.path == "/api/dataset":
                    return self._send(200, app.dataset(q.get("model", app.model)))
                if u.path == "/api/probe":
                    return self._send(200, app.probe_static())
                if u.path == "/img":
                    p = (ROOT / q["p"]).resolve()
                    if (ROOT / "data") not in p.parents or not p.exists():
                        return self._send(404, b"no", "text/plain")
                    ct = "image/png" if p.suffix == ".png" else "image/jpeg"
                    return self._send(200, p.read_bytes(), ct)
                return self._send(404, b"not found", "text/plain")
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if u.path == "/api/load":
                    app.load(body["model"])
                    return self._send(200, {"model": app.model, "curated": app.curated_items(app.model)})
                if u.path == "/api/ask":
                    return self._send(
                        200,
                        app.ask(body["item"], body.get("qid"), body.get("custom"), body.get("order", 0)),
                    )
                if u.path == "/api/probe_live":
                    return self._send(200, app.probe_live(body["item"]))
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-vl-3b", choices=LOCAL_MODELS)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--port", type=int, default=8810)
    a = ap.parse_args()
    if not a.replay:
        try:
            import torch

            if not torch.cuda.is_available():
                print("no CUDA GPU found: falling back to --replay", flush=True)
                a.replay = True
        except ImportError:
            a.replay = True
    app = App(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(app))
    print(
        f"demo ready: http://127.0.0.1:{a.port}  (mode: {'replay' if a.replay else 'live'}, model: {app.model})",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
