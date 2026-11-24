"""Evaluate one model and save a resumable JSONL log plus Parquet results."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict

import pandas as pd
from tqdm import tqdm

from backends import Mock, QuotaExhausted, make_backend
from common import (
    PROC,
    RESULTS,
    ROOT,
    assign_subsets,
    build_queries,
    config_hash,
    load_config,
    load_model_image,
)


def load_run_items(datasets: list[str], audited_only: bool) -> pd.DataFrame:
    audited_path = PROC / "real" / "items_real_audited.parquet"
    real_path = audited_path if audited_path.exists() else PROC / "real" / "items_real.parquet"
    synthetic_path = PROC / "synthetic" / "items_synthetic.parquet"
    sources = (
        ("real", real_path, "build_real_dataset.py"),
        ("synthetic", synthetic_path, "generate_synthetic.py"),
    )
    frames = []
    for dataset, path, builder in sources:
        if dataset not in datasets:
            continue
        if not path.exists():
            print(f"[warn] {path} missing; run {builder}")
            continue
        items = pd.read_parquet(path)
        if dataset == "real" and audited_only and "audit_keep" in items:
            items = items[items["audit_keep"].eq(True)]
        frames.append(items)
    if not frames:
        raise SystemExit("no items")
    return pd.concat(frames, ignore_index=True)


def runtime_versions():
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
    ap.add_argument(
        "--limit", type=int, default=None, help="items per dataset (first N after a fixed shuffle)"
    )
    ap.add_argument("--audited-only", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the results file name")
    a = ap.parse_args()

    cfg = load_config()
    spec = cfg["models"][a.model]
    items = load_run_items(a.datasets.split(","), a.audited_only)
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
    print(
        f"{a.model}: {len(recs)} items, {len(queries)} queries ({len(queries) / max(1, len(recs)):.1f}/item), "
        f"{len(done)} done, {len(todo)} to run"
    )

    t_load = time.time()
    backend = make_backend(spec, cfg["runtime"])
    t_load = time.time() - t_load
    image_cache = {}
    max_side = spec.get("max_image_side", cfg["runtime"]["max_image_side"])
    lat = []
    stopped = None
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for q in tqdm(todo, desc=a.model):
            if q.image_path not in image_cache:
                if len(image_cache) > 64:
                    image_cache.clear()
                image_cache[q.image_path] = load_model_image(ROOT / q.image_path, max_side)
            t0 = time.time()
            err = None
            try:
                if isinstance(backend, Mock):
                    r = backend.ask_query(q, by_item[q.item_id])
                else:
                    r = backend.ask(image_cache[q.image_path], q.prompt, q.options, q.system)
            except QuotaExhausted as e:
                stopped = str(e)[:300]
                break
            except Exception as e:  # noqa: BLE001  (log and continue; the row is retried on resume)
                r = {
                    "answer": None,
                    "p_options": None,
                    "logits": None,
                    "option_mass": None,
                    "raw_text": None,
                }
                err = f"{type(e).__name__}: {e}"[:500]
            dt = time.time() - t0
            lat.append(dt)
            probabilities = r["p_options"] or {}
            logits = r["logits"]
            logit_diff = logits[q.correct] - logits[q.distractor] if logits and q.distractor else None
            query_record = asdict(q)
            prompt = query_record.pop("prompt")
            row = {
                **query_record,
                "prompt": prompt,
                "model": a.model,
                **r,
                "p_options": probabilities,
                "p_correct": probabilities.get(q.correct),
                "p_distractor": probabilities.get(q.distractor) if q.distractor else None,
                "logit_diff": logit_diff,
                "readout": "logits" if logits else ("text" if r["answer"] else None),
                "latency_s": round(dt, 4),
                "error": err,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            if err and len(lat) < 5:
                print("error:", err)

    with out_jsonl.open(encoding="utf-8") as log:
        rows = [json.loads(line) for line in log if line.strip()]
    df = pd.DataFrame(rows)
    df = df.sort_values("error", na_position="last").drop_duplicates("qid", keep="first") if len(df) else df
    for c in ("p_options", "logits", "options"):
        if c in df:
            df[c] = df[c].apply(json.dumps)
    df.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    meta = {
        "model": a.model,
        "spec": spec,
        "config_hash": config_hash(),
        "profile": a.profile,
        "blocks": sorted(blocks),
        "n_items": len(recs),
        "n_queries": len(queries),
        "n_rows": len(df),
        "n_errors": int(df["error"].notna().sum()) if len(df) else 0,
        "load_s": round(t_load, 1),
        "mean_latency_s": round(sum(lat) / len(lat), 4) if lat else None,
        "versions": runtime_versions(),
        "prompts": cfg["prompts"],
        "stopped_early": stopped,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (RESULTS / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: meta[k] for k in ("n_rows", "n_errors", "mean_latency_s", "stopped_early")}))
    if lat:
        per_item = sum(lat) / len(lat) * len(queries) / max(1, len(recs))
        print(f"~{per_item:.1f} s/item -> 1,200 items ~ {per_item * 1200 / 3600:.1f} h")
    if stopped:
        print("[quota] stopped early; rerun the same command later to resume:", stopped)


if __name__ == "__main__":
    main()
