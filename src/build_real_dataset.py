"""Apply the Contradiction Validity Conditions (CVC1-4) and build the real subset.

Two sources, one rule set:
  * V-COCO (COCO 2014): every person is boxed by COCO, every V-COCO person is labelled for all actions.
  * HICO-DET train (HAKE instance-level release): full multi-pair annotations.

Stages: annotations (CVC1a/b, CVC2a, CVC3, CVC4) -> download candidate JPEGs -> detector at a low
threshold (CVC1c: no unannotated person; CVC2b: union object count == 1) -> marker rendering.

Outputs: data/processed/real/{items_real.parquet, funnel.csv, candidates_rejected.csv, detections.json, images/}
"""

from __future__ import annotations

import argparse
import json
import math
import random
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from common import (
    PROC,
    RAW,
    ROOT,
    area,
    cluster_boxes,
    config_hash,
    download_hake_annotations,
    draw_markers,
    ensure_dirs,
    iou,
    load_config,
    load_hoi_list,
    resize_to_fit,
)

OUT = PROC / "real"
IMG_RAW = RAW / "real_images"


class Funnel:
    """Counts of *images* surviving each stage, per source."""

    def __init__(self):
        self.counts_by_stage = defaultdict(Counter)

    def count_pass(self, source, stage, count=1):
        self.counts_by_stage[stage][source] += count

    def to_frame(self):
        sources = sorted({source for counts in self.counts_by_stage.values() for source in counts})
        rows = []
        for stage, counts in self.counts_by_stage.items():
            rows.append(
                {"stage": stage, **{source: counts[source] for source in sources}, "total": counts.total()}
            )
        return pd.DataFrame(rows)


def xywh_to_xyxy(box):
    x, y, width, height = box
    return [x, y, x + width, y + height]


def passes_person_checks(funnel, src, W, H, boxes, thresholds):
    """CVC1a/b on the annotated person boxes. Returns True if the image survives."""
    if len(boxes) != 2:
        return False
    funnel.count_pass(src, "CVC1a: exactly 2 persons (annotation)")
    if (
        iou(boxes[0], boxes[1]) >= thresholds["max_person_iou"]
        or min(area(b) for b in boxes) / (W * H) < thresholds["relaxed_person_area_frac"]
    ):
        return False
    funnel.count_pass(src, "CVC1b: separable (IoU<0.3), each >= 2% area")
    return True



def fetch_vcoco(cfg):
    r = cfg["real"]
    vd, cd = RAW / "vcoco", RAW / "coco" / "annotations"
    vd.mkdir(parents=True, exist_ok=True)
    for k, url in r["vcoco_urls"].items():
        p = vd / f"vcoco_{k}.json"
        if not p.exists():
            p.write_bytes(requests.get(url, timeout=120).content)
    if not (cd / "instances_train2014.json").exists():
        z = RAW / "coco" / "annotations_trainval2014.zip"
        if not z.exists():
            with requests.get(r["coco_ann_zip"], stream=True, timeout=600) as resp, open(z, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
        with zipfile.ZipFile(z) as zf:
            for n in ("annotations/instances_train2014.json", "annotations/instances_val2014.json"):
                zf.extract(n, RAW / "coco")


def vcoco_candidates(cfg, funnel):
    fetch_vcoco(cfg)
    r, thresholds = cfg["real"], cfg["real"]["thresholds"]
    wl = r["whitelist_vcoco"]
    allowed_actions = {rule["verb"] for rule in wl}
    cats, anns, imgs = {}, defaultdict(list), {}
    for s in ("train", "val"):
        d = json.loads((RAW / "coco" / "annotations" / f"instances_{s}2014.json").read_text())
        cats.update({c["id"]: c["name"] for c in d["categories"]})
        for i in d["images"]:
            imgs[i["id"]] = (s, i["width"], i["height"], i["file_name"])
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
    byid = {a["id"]: a for L in anns.values() for a in L}
    labels: dict[int, dict[str, list]] = defaultdict(
        dict
    )  # person ann id -> action -> [(role, obj ann id)]
    pimg = {}
    for f in ("trainval", "test"):
        for act in json.loads((RAW / "vcoco" / f"vcoco_{f}.json").read_text()):
            N, R = len(act["ann_id"]), len(act["role_name"])
            roles = [act["role_object_id"][k * N : (k + 1) * N] for k in range(R)]
            for k in range(N):
                pid = act["ann_id"][k]
                pimg[pid] = act["image_id"][k]
                labels[pid]  # touch: every V-COCO person gets an entry even with no positive action
                if act["label"][k]:
                    labels[pid][act["action_name"]] = [
                        (act["role_name"][j], roles[j][k]) for j in range(1, R)
                    ]
    img_persons = defaultdict(set)
    for pid, im in pimg.items():
        img_persons[im].add(pid)

    out = []
    src = "vcoco"
    for im, vp in tqdm(img_persons.items(), desc="V-COCO CVC1-4"):
        split, W, H, fname = imgs[im]
        funnel.count_pass(src, "all images")
        persons = [a for a in anns[im] if cats[a["category_id"]] == "person" and not a["iscrowd"]]
        has_wl = any(allowed_actions.intersection(labels[person_id]) for person_id in vp)
        if not has_wl:
            continue
        funnel.count_pass(src, "has whitelisted action")
        pboxes = [xywh_to_xyxy(p["bbox"]) for p in persons]
        if not passes_person_checks(funnel, src, W, H, pboxes, thresholds):
            continue
        if not all(p["id"] in vp for p in persons):
            continue  # both people must carry exhaustive V-COCO labels
        funnel.count_pass(src, "(V-COCO only) both persons action-labelled")
        accepted = None
        stage_hits = set()
        for w in wl:
            for ai in (0, 1):
                a, o = persons[ai]["id"], persons[1 - ai]["id"]
                roles = labels[a].get(w["verb"])
                if not roles:
                    continue
                oid = next((oo for rn, oo in roles if rn == w["role"]), -1)
                if oid in (-1, None) or oid not in byid or cats[byid[oid]["category_id"]] != w["object"]:
                    continue
                n_cls = sum(1 for x in anns[im] if cats[x["category_id"]] == w["object"])
                if n_cls != 1:
                    continue
                stage_hits.add("CVC2a")
                if w["verb"] in labels[o]:
                    continue  # CVC3: the other person also does the action
                stage_hits.add("CVC3")
                other_links = {ac for ac, rr in labels[o].items() for (_, oo) in rr if oo == oid}
                if other_links - set(w["allowed_other"]):
                    continue  # CVC4
                accepted = dict(
                    w=w, ai=ai, obj_box=xywh_to_xyxy(byid[oid]["bbox"]), other=sorted(other_links)
                )
                break
            if accepted:
                break
        if "CVC2a" in stage_hits:
            funnel.count_pass(src, "CVC2a: one target object (annotation)")
        if "CVC3" in stage_hits:
            funnel.count_pass(src, "CVC3: action linked to exactly one person")
        if not accepted:
            continue
        funnel.count_pass(src, "CVC4: distractor has no conflicting label")
        ai = accepted["ai"]
        out.append(
            dict(
                source=src,
                file=fname,
                url=r["coco_image_url"].format(split=split, id=im),
                W=W,
                H=H,
                w=accepted["w"],
                actor_box=pboxes[ai],
                distractor_box=pboxes[1 - ai],
                obj_box=accepted["obj_box"],
                distractor_labels=";".join(accepted["other"]),
            )
        )
    return out



def hico_candidates(cfg, funnel):
    r, thresholds = cfg["real"], cfg["real"]["thresholds"]
    annotation_path, hoi_path = download_hake_annotations(cfg)
    ann = json.loads(annotation_path.read_text())
    hoi = load_hoi_list(hoi_path)
    wl = r["whitelist_hico"]
    wl_pairs = {(w["object"], w["verb"]) for w in wl}
    src = "hico"
    out = []
    for fn, rec in tqdm(ann.items(), desc="HICO CVC1-4"):
        funnel.count_pass(src, "all images")
        rows = []
        for l in rec["labels"]:
            if not l.get("human_bbox"):
                continue
            o, v = hoi[l["hoi_id"]]
            rows.append({"obj": o, "verb": v, "hb": l["human_bbox"], "ob": l.get("object_bbox") or None})
        annotated_pairs = {(row["obj"], row["verb"]) for row in rows}
        if not annotated_pairs.intersection(wl_pairs):
            continue
        funnel.count_pass(src, "has whitelisted action")
        W, H = rec["labels"][0]["width"], rec["labels"][0]["height"]
        plab, preps = cluster_boxes([x["hb"] for x in rows], thresholds["cluster_iou"])
        if not passes_person_checks(funnel, src, W, H, preps, thresholds):
            continue
        accepted, stage_hits = None, set()
        for w in wl:
            if (w["object"], w["verb"]) not in annotated_pairs:
                continue
            _, oreps = cluster_boxes(
                [x["ob"] for x in rows if x["obj"] == w["object"] and x["ob"]], thresholds["cluster_iou"]
            )
            if len(oreps) != 1:
                continue
            stage_hits.add("CVC2a")
            actors = {
                plab[i] for i, x in enumerate(rows) if (x["obj"], x["verb"]) == (w["object"], w["verb"])
            }
            if len(actors) != 1:
                continue
            stage_hits.add("CVC3")
            a = actors.pop()
            other = {x["verb"] for i, x in enumerate(rows) if plab[i] != a and x["obj"] == w["object"]}
            if other - set(w["allowed_other"]):
                continue
            accepted = dict(w=w, ai=a, obj_box=oreps[0], other=sorted(other))
            break
        if "CVC2a" in stage_hits:
            funnel.count_pass(src, "CVC2a: one target object (annotation)")
        if "CVC3" in stage_hits:
            funnel.count_pass(src, "CVC3: action linked to exactly one person")
        if not accepted:
            continue
        funnel.count_pass(src, "CVC4: distractor has no conflicting label")
        ai = accepted["ai"]
        w = dict(accepted["w"])
        w["object"] = w["object"].replace("_", " ")  # COCO naming, shared with the detector
        out.append(
            dict(
                source=src,
                file=fn,
                url=None,
                W=W,
                H=H,
                w=w,
                actor_box=preps[ai],
                distractor_box=preps[1 - ai],
                obj_box=accepted["obj_box"],
                distractor_labels=";".join(accepted["other"]),
            )
        )
    return out



def download_images(cfg, cands):
    from huggingface_hub import hf_hub_download, list_repo_files

    IMG_RAW.mkdir(parents=True, exist_ok=True)
    idx_path = RAW / "hico_image_index.json"
    index = None

    def hico_index():
        nonlocal index
        if index is None:
            if idx_path.exists():
                index = json.loads(idx_path.read_text())
            else:
                index = {
                    Path(p).name: p
                    for p in list_repo_files(cfg["real"]["image_repo"], repo_type="dataset")
                    if p.endswith(".jpg")
                }
                idx_path.write_text(json.dumps(index))
        return index

    if any(c["source"] == "hico" for c in cands):
        hico_index()

    def download_image(c):
        dst = IMG_RAW / c["file"]
        if dst.exists() and dst.stat().st_size > 0:
            return True
        try:
            if c["source"] == "vcoco":
                resp = requests.get(c["url"], timeout=60)
                resp.raise_for_status()
                dst.write_bytes(resp.content)
            else:
                if c["file"] not in index:
                    return False
                p = hf_hub_download(
                    cfg["real"]["image_repo"],
                    index[c["file"]],
                    repo_type="dataset",
                    local_dir=RAW / "_hf_tmp",
                )
                Path(p).replace(dst)
            Image.open(dst).verify()
            return True
        except Exception as e:  # noqa: BLE001
            print("download failed", c["file"], e)
            return False

    with ThreadPoolExecutor(12) as ex:
        ok = list(tqdm(ex.map(download_image, cands), total=len(cands), desc="download"))
    return [c for c, good in zip(cands, ok) if good]



def detect_objects(cfg, files, cache_path):
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [f for f in files if f not in cache]
    if todo:
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        name = cfg["real"]["detector"]
        proc = AutoImageProcessor.from_pretrained(name)
        model = AutoModelForObjectDetection.from_pretrained(name).to(dev).eval()
        id2label = model.config.id2label
        for k, fn in enumerate(tqdm(todo, desc="detector")):
            im = Image.open(IMG_RAW / fn).convert("RGB")
            with torch.no_grad():
                inp = proc(images=im, return_tensors="pt").to(dev)
                out = model(**inp)
            res = proc.post_process_object_detection(out, threshold=0.1, target_sizes=[im.size[::-1]])[0]
            cache[fn] = [
                {"label": id2label[l], "score": round(s, 4), "box": [round(v, 1) for v in b]}
                for s, l, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist())
            ]
            if k % 100 == 0:
                cache_path.write_text(json.dumps(cache))
        cache_path.write_text(json.dumps(cache))
    return cache


def check_detections(cfg, c, dets):
    thresholds = cfg["real"]["thresholds"]
    WH = c["W"] * c["H"]
    persons = [c["actor_box"], c["distractor_box"]]
    extra = [
        d
        for d in dets
        if d["label"] == "person"
        and d["score"] >= thresholds["det_person_conf"]
        and area(d["box"]) / WH >= thresholds["det_person_min_area_frac"]
        and max(iou(d["box"], p) for p in persons) < thresholds["det_match_iou"]
    ]
    target = c["w"]["object"]
    objs = sorted(
        [d for d in dets if d["label"] == target and d["score"] >= thresholds["det_object_conf"]],
        key=lambda d: -d["score"],
    )
    union = [c["obj_box"]]
    for d in objs:
        if max(iou(d["box"], u) for u in union) < thresholds["det_match_iou"]:
            union.append(d["box"])
    return dict(
        n_extra_people=len(extra),
        n_objects_union=len(union),
        obj_detected=any(iou(d["box"], c["obj_box"]) >= 0.3 for d in objs),
        extra_person_scores=[d["score"] for d in extra],
        extra_object_scores=[
            d["score"] for d in objs if iou(d["box"], c["obj_box"]) < thresholds["det_match_iou"]
        ],
    )



def render_items(cfg, cands, max_side):
    rng = random.Random(cfg["seed"])
    thresholds = cfg["real"]["thresholds"]
    all_wl = cfg["real"]["whitelist_vcoco"] + cfg["real"]["whitelist_hico"]
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in sorted(cands, key=lambda c: (c["source"], c["file"])):
        w = c["w"]
        im = Image.open(IMG_RAW / c["file"]).convert("RGB")
        s = min(1.0, max_side / max(im.size))
        im = resize_to_fit(im, max_side)
        ab, db, ob = ([v * s for v in b] for b in (c["actor_box"], c["distractor_box"], c["obj_box"]))
        actor_letter = rng.choice("AB")
        distr_letter = "B" if actor_letter == "A" else "A"
        acx, dcx = (ab[0] + ab[2]) / 2, (db[0] + db[2]) / 2
        actor_side = "left" if acx < dcx else "right"
        W, H = im.size
        lr_ok = abs(acx - dcx) >= thresholds["lr_min_center_gap_frac"] * W
        item_id = f"{c['source']}_{Path(c['file']).stem.split('_')[-1]}"
        mp, pp = img_dir / f"{item_id}_marked.jpg", img_dir / f"{item_id}_plain.jpg"
        draw_markers(im, {actor_letter: ab, distr_letter: db}).save(mp, quality=92)
        im.save(pp, quality=92)
        ctr = lambda b: ((b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H)  # noqa: E731
        (ax, ay), (dx, dy), (ox, oy) = ctr(ab), ctr(db), ctr(ob)
        foil = rng.choice(
            [
                x
                for x in all_wl
                if x["object"].replace("_", " ") != w["object"] and x["q2_obj"] != w["q2_obj"]
            ]
        )
        a_area, d_area = area(ab) / (W * H), area(db) / (W * H)
        rows.append(
            dict(
                item_id=item_id,
                dataset="real",
                source=c["source"],
                source_file=c["file"],
                object=w["object"],
                verb=w["verb"],
                verb_key=f"{w['verb']} {w['object']}",
                ing=w["ing"],
                obj_np=w["obj_np"],
                q2=f"{w['ing']} {w['q2_obj']}",
                q2_foil=f"{foil['ing']} {foil['q2_obj']}",
                actor_letter=actor_letter,
                actor_side=actor_side,
                lr_ok=bool(lr_ok),
                image_marked=mp.relative_to(ROOT).as_posix(),
                image_plain=pp.relative_to(ROOT).as_posix(),
                actor_box=ab,
                distractor_box=db,
                obj_box=ob,
                img_w=W,
                img_h=H,
                actor_area=a_area,
                distractor_area=d_area,
                min_person_area=min(a_area, d_area),
                person_iou=iou(ab, db),
                obj_area=area(ob) / (W * H),
                person_dist=((ax - dx) ** 2 + (ay - dy) ** 2) ** 0.5,
                obj_actor_dist=((ax - ox) ** 2 + (ay - oy) ** 2) ** 0.5,
                obj_distractor_dist=((dx - ox) ** 2 + (dy - oy) ** 2) ** 0.5,
                log_area_ratio=math.log(a_area / d_area),
                distractor_labels=c["distractor_labels"],
                obj_detected=c["obj_detected"],
                audit_status="pending",
            )
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="vcoco,hico")
    a = ap.parse_args()
    cfg = load_config()
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    thresholds = cfg["real"]["thresholds"]
    funnel = Funnel()
    cands = []
    if "vcoco" in a.sources:
        cands += vcoco_candidates(cfg, funnel)
    if "hico" in a.sources:
        cands += hico_candidates(cfg, funnel)
    print(f"annotation-stage candidates: {len(cands)}")

    cands = download_images(cfg, cands)
    for c in cands:
        funnel.count_pass(c["source"], "image downloaded")
    dets = detect_objects(cfg, [c["file"] for c in cands], OUT / "detections.json")
    rejected, passed = [], []
    for c in cands:
        c.update(check_detections(cfg, c, dets[c["file"]]))
        if c["n_extra_people"] > 0:
            rejected.append({**c, "reason": "CVC1c extra person"})
            continue
        funnel.count_pass(c["source"], "CVC1c: no unannotated person (detector, conf>=0.4)")
        if c["n_objects_union"] != 1:
            rejected.append({**c, "reason": "CVC2b second object"})
            continue
        funnel.count_pass(c["source"], "CVC2b: one object (annotation U detector, conf>=0.2)")
        passed.append(c)

    strict = [
        c
        for c in passed
        if min(area(c["actor_box"]), area(c["distractor_box"])) / (c["W"] * c["H"])
        >= thresholds["min_person_area_frac"]
    ]
    if len(strict) >= thresholds["min_items"]:
        final, thr = strict, thresholds["min_person_area_frac"]
    else:
        final, thr = passed, thresholds["relaxed_person_area_frac"]
    for c in strict:
        funnel.count_pass(c["source"], "CVC1 strict: each person >= 3% area")
    for c in final:
        funnel.count_pass(c["source"], f"FINAL automatic candidates (area >= {thr:.0%})")

    df = render_items(cfg, final, cfg["runtime"]["max_image_side"])
    df["person_area_threshold"] = thr
    df["config_hash"] = config_hash()
    df.to_parquet(OUT / "items_real.parquet", index=False)
    pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ("w",)}
            | {"verb_key": f"{r['w']['verb']} {r['w']['object']}"}
            for r in rejected
        ]
    ).to_csv(OUT / "candidates_rejected.csv", index=False)
    ft = funnel.to_frame()
    ft.to_csv(OUT / "funnel.csv", index=False)
    print(ft.to_string(index=False))
    print(df.groupby(["source", "verb_key"]).size().to_string())
    if len(final) < thresholds["min_items"]:
        print(
            f"[note] {len(final)} items < min_items={thresholds['min_items']}: area relaxed to {thr:.0%}; "
            "the synthetic subset carries the statistical power (PLAN 2b)."
        )


if __name__ == "__main__":
    main()
