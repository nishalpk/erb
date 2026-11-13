"""D3/D5: run one model over all items x conditions x questions x option orders.

    python src/04_run_models.py --model smolvlm2-2.2b --limit 50          # D3 smoke test + timing
    python src/04_run_models.py --model qwen2.5-vl-3b                      # full run (resumable)
    python src/04_run_models.py --model gemini-2.5-flash --profile core --blocks main --limit 150

Results go to results/<model>.jsonl (one line per query, appended, resumable) and are consolidated into
results/<model>.parquet plus results/<model>.meta.json (versions, config hash, templates, timing).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from backends import Mock, QuotaExhausted, make_backend  # noqa: E402
from common import PROC, RESULTS, ROOT, assign_subsets, build_queries, config_hash, load_config, resize_max  # noqa: E402


def load_items(datasets: list[str], audited_only: bool) -> pd.DataFrame:
    frames = []
    if "real" in datasets:
        p_aud = PROC / "real" / "items_real_audited.parquet"
        p = p_aud if p_aud.exists() else PROC / "real" / "items_real.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if audited_only and "audit_keep" in df:
                df = df[df["audit_keep"] == True]  # noqa: E712
            frames.append(df)
        else:
            print(f"[warn] {p} missing; run 01_build_real.py")
    if "synthetic" in datasets:
        p = PROC / "synthetic" / "items_synthetic.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            print(f"[warn] {p} missing; run 03_make_synthetic.py")
    if not frames:
        raise SystemExit("no items")
    return pd.concat(frames, ignore_index=True)


def versions():
    v = {"python": platform.python_version()}
    for m in ("torch", "transformers", "accelerate", "bitsandbytes", "google.genai"):
        try:
            mod = __import__(m, fromlist=["__version__"])
            v[m] = getattr(mod, "__version__", "?")
        except ImportError:
            pass
    try:
        import torch
        v["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", default="real,synthetic")
    ap.add_argument("--profile", default="full", choices=["full", "core"])
    ap.add_argument("--blocks", default="main,robust,ablation,lr")
    ap.add_argument("--limit", type=int, default=None, help="items per dataset (first N after a fixed shuffle)")
    ap.add_argument("--audited-only", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the results file name")
    a = ap.parse_args()

    cfg = load_config()
    spec = cfg["models"][a.model]
    items = load_items(a.datasets.split(","), a.audited_only)
    items = items.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
    if a.limit:
        items = items.groupby("dataset", group_keys=False).head(a.limit).reset_index(drop=True)
    recs = items.to_dict("records")
    recs = assign_subsets(recs, cfg, cfg["seed"])
    blocks = set(a.blocks.split(","))
    queries = [q for it in recs for q in build_queries(cfg, it, a.profile, blocks)]
    by_item = {it["item_id"]: it for it in recs}

    RESULTS.mkdir(exist_ok=True)
    stem = a.model + (f"_{a.tag}" if a.tag else "")
    out_jsonl = RESULTS / f"{stem}.jsonl"
    done = set()
    if out_jsonl.exists():
        with open(out_jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("error") is None:
                        done.add(r["qid"])
                except json.JSONDecodeError:
                    pass
    todo = [q for q in queries if q.qid not in done]
    print(f"{a.model}: {len(recs)} items, {len(queries)} queries ({len(queries) / max(1, len(recs)):.1f}/item), "
          f"{len(done)} done, {len(todo)} to run")

    t_load = time.time()
    backend = make_backend(spec, cfg["runtime"])
    t_load = time.time() - t_load
    img_cache: dict[str, Image.Image] = {}
    max_side = spec.get("max_image_side", cfg["runtime"]["max_image_side"])
    lat = []
    stopped = None
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for q in tqdm(todo, desc=a.model):
            if q.image_path not in img_cache:
                if len(img_cache) > 64:
                    img_cache.clear()
                img_cache[q.image_path] = resize_max(Image.open(ROOT / q.image_path).convert("RGB"), max_side)
            t0 = time.time()
            err = None
            try:
                if isinstance(backend, Mock):
                    r = backend.ask_query(q, by_item[q.item_id])
                else:
                    r = backend.ask(img_cache[q.image_path], q.prompt, q.options, q.system)
            except QuotaExhausted as e:
                stopped = str(e)[:300]
                break
            except Exception as e:  # noqa: BLE001  (log and continue; the row is retried on resume)
                r = {"answer": None, "p_options": None, "logits": None, "option_mass": None, "raw_text": None}
                err = f"{type(e).__name__}: {e}"[:500]
            dt = time.time() - t0
            lat.append(dt)
            p = r["p_options"] or {}
            row = {**{k: v for k, v in asdict(q).items() if k not in ("prompt",)}, "prompt": q.prompt,
                   "model": a.model, "answer": r["answer"], "p_options": p, "logits": r["logits"],
                   "option_mass": r["option_mass"], "raw_text": r["raw_text"],
                   "p_correct": p.get(q.correct), "p_distractor": p.get(q.distractor) if q.distractor else None,
                   "logit_diff": (r["logits"][q.correct] - r["logits"][q.distractor]) if (r["logits"] and q.distractor) else None,
                   "readout": "logits" if r["logits"] else ("text" if r["answer"] else None),
                   "latency_s": round(dt, 4), "error": err}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if err and len(lat) < 5:
                print("error:", err)

    # consolidate (last successful row per qid wins)
    rows = [json.loads(l) for l in open(out_jsonl, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    df = df.sort_values("error", na_position="last").drop_duplicates("qid", keep="first") if len(df) else df
    for c in ("p_options", "logits", "options"):
        if c in df:
            df[c] = df[c].apply(json.dumps)
    df.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    meta = {"model": a.model, "spec": spec, "config_hash": config_hash(), "profile": a.profile, "blocks": sorted(blocks),
            "n_items": len(recs), "n_queries": len(queries), "n_rows": len(df),
            "n_errors": int(df["error"].notna().sum()) if len(df) else 0,
            "load_s": round(t_load, 1), "mean_latency_s": round(sum(lat) / len(lat), 4) if lat else None,
            "versions": versions(), "prompts": cfg["prompts"], "stopped_early": stopped,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (RESULTS / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: meta[k] for k in ("n_rows", "n_errors", "mean_latency_s", "stopped_early")}))
    if lat:
        per_item = sum(lat) / len(lat) * len(queries) / max(1, len(recs))
        print(f"~{per_item:.1f} s/item -> 1,200 items ~ {per_item * 1200 / 3600:.1f} h")
    if stopped:
        print("[quota] stopped early; rerun the same command later to resume:", stopped)


if __name__ == "__main__":
    main()
