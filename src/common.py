"""Shared helpers: paths, config, box maths, marker rendering, query construction."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path=None):
    """Read KEY=VALUE lines from the git-ignored .env into os.environ (existing variables win)."""
    import os
    p = Path(path or ROOT / ".env")
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip() and not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip()


load_dotenv()
CONFIG_PATH = ROOT / "configs" / "verbs.yaml"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash() -> str:
    return hashlib.sha1(CONFIG_PATH.read_bytes()).hexdigest()[:10]


def ensure_dirs():
    for p in (RAW, PROC, RESULTS, REPORTS, FIG):
        p.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ boxes (xyxy)
def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = area(a) + area(b) - inter
    return inter / u if u > 0 else 0.0


def cluster_boxes(boxes, thr):
    """Greedy single-link clustering; returns list of cluster index per box and cluster mean boxes."""
    labels = [-1] * len(boxes)
    reps: list[list[float]] = []
    members: list[list[int]] = []
    for i, b in enumerate(boxes):
        best, best_iou = -1, thr
        for c, r in enumerate(reps):
            v = max(iou(b, boxes[j]) for j in members[c])
            if v >= best_iou:
                best, best_iou = c, v
        if best < 0:
            reps.append(list(b))
            members.append([i])
            labels[i] = len(reps) - 1
        else:
            members[best].append(i)
            labels[i] = best
            ms = [boxes[j] for j in members[best]]
            reps[best] = [sum(m[k] for m in ms) / len(ms) for k in range(4)]
    return labels, reps


# ------------------------------------------------------------------ marker overlay
def _font(size):
    for name in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_markers(img: Image.Image, boxes: dict[str, list[float]]) -> Image.Image:
    """Set-of-marks overlay: identical style for every marker (no colour confound)."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    lw = max(2, round(min(W, H) / 160))
    fs = max(16, round(min(W, H) / 14))
    font = _font(fs)
    for letter, b in boxes.items():
        x1, y1, x2, y2 = [float(v) for v in b]
        d.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=lw + 2)
        d.rectangle([x1, y1, x2, y2], outline=(255, 221, 0), width=lw)
        tw = d.textlength(letter, font=font)
        pad = fs * 0.2
        th = fs + 2 * pad
        tx = min(max(0, x1), W - tw - 2 * pad)
        ty = y1 - th if y1 - th >= 0 else y1  # tag sits above the box (keeps the head visible) when it fits
        ty = min(max(0, ty), H - th)
        d.rectangle([tx, ty, tx + tw + 2 * pad, ty + fs + 2 * pad], fill=(255, 221, 0), outline=(0, 0, 0), width=2)
        d.text((tx + pad, ty + pad * 0.6), letter, fill=(0, 0, 0), font=font)
    return im


def resize_max(img: Image.Image, max_side: int) -> Image.Image:
    W, H = img.size
    s = max_side / max(W, H)
    if s >= 1:
        return img
    return img.resize((round(W * s), round(H * s)), Image.LANCZOS)


# ------------------------------------------------------------------ queries
@dataclass
class Query:
    qid: str
    item_id: str
    dataset: str
    question: str        # q1 q2 q2_foil q3 q4
    condition: str       # none neutral consistent contra_1..4
    block: str           # main | robust | ablation | lr
    scheme: str          # marker | lr
    order: int           # 0 = options as (O1,O2)=(A,B)/(left,right); 1 = reversed
    cap_tpl: int
    q_tpl: int
    image_path: str
    system: str | None
    prompt: str
    options: list[str]
    correct: str         # option string that is correct
    distractor: str | None  # option string naming the distractor (q4 only)


def _ref(scheme, actor_letter, actor_side, which):
    """Entity reference string for 'actor' or 'distractor'."""
    if scheme == "marker":
        other = "B" if actor_letter == "A" else "A"
        return actor_letter if which == "actor" else other
    other = "right" if actor_side == "left" else "left"
    side = actor_side if which == "actor" else other
    return f"the person on the {side}"


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def make_caption(cfg, cond, item, scheme, tpl=0):
    P = cfg["prompts"]
    if cond == "none":
        return None
    ing, obj = item["ing"], item["obj_np"]
    if cond == "neutral":
        x1 = _ref(scheme, "A", "left", "actor") if scheme == "marker" else "the person on the left"
        x2 = _ref(scheme, "A", "left", "distractor") if scheme == "marker" else "the person on the right"
        return P["captions"]["neutral"].format(X1=x1, X2=x2)
    who = "actor" if cond == "consistent" else "distractor"
    X = _ref(scheme, item["actor_letter"], item["actor_side"], who)
    tmpl = P["caption_paraphrases"][tpl] if (tpl and cond == "contra_2") else P["captions"][cond]
    return _cap(tmpl.format(X=X, ing=ing, obj=obj).replace("  ", " ")).replace(". the", ". The")


def build_queries(cfg, item: dict, profile: str, blocks: set[str]) -> list[Query]:
    P = cfg["prompts"]
    prof = cfg["profiles"][profile]
    out: list[Query] = []
    actor = item["actor_letter"]
    distr = "B" if actor == "A" else "A"

    def add(question, cond, block, scheme, order, cap_tpl, q_tpl, text, options, correct, distractor, system=None, img_key="image_marked"):
        qid = f"{item['item_id']}|{question}|{cond}|{block}|{scheme}|o{order}|c{cap_tpl}|q{q_tpl}"
        out.append(Query(qid, item["item_id"], item["dataset"], question, cond, block, scheme, order, cap_tpl, q_tpl,
                         item[img_key], system, text, options, correct, distractor))

    def with_caption(cond, scheme, body, cap_tpl=0):
        cap = make_caption(cfg, cond, item, scheme, cap_tpl)
        return body if cap is None else P["caption_wrapper"].format(caption=cap) + "\n" + body

    def q4(cond, block, scheme="marker", cap_tpl=0, q_tpl=0, system=None):
        for order in (0, 1):
            if scheme == "marker":
                o = ["A", "B"] if order == 0 else ["B", "A"]
                suffix = P["q4_suffix_marker"]
                corr, dis = actor, distr
                img_key = "image_marked"
            else:
                o = ["left", "right"] if order == 0 else ["right", "left"]
                suffix = P["q4_suffix_lr"]
                corr = item["actor_side"]
                dis = "right" if corr == "left" else "left"
                img_key = "image_plain"
            if scheme == "lr":
                O1, O2 = f"the person on the {o[0]}", f"the person on the {o[1]}"
            else:
                O1, O2 = o
            body = P["q4"][q_tpl].format(ing=item["ing"], obj=item["obj_np"], O1=O1, O2=O2) + " " + suffix.format(O1=o[0], O2=o[1])
            add("q4", cond, block, scheme, order, cap_tpl, q_tpl, with_caption(cond, scheme, body, cap_tpl), o, corr, dis, system, img_key)

    # ---- main block
    if "main" in blocks:
        for c in prof["q4_conditions"]:
            q4(c, "main")
        for c in prof["q2_conditions"]:
            body = P["q2"].format(q2=item["q2"])
            add("q2", c, "main", "marker", 0, 0, 0, with_caption(c, "marker", body), ["Yes", "No"], "Yes", None)
        if prof.get("q2_foil"):
            body = P["q2"].format(q2=item["q2_foil"])
            add("q2_foil", "none", "main", "marker", 0, 0, 0, body, ["Yes", "No"], "No", None)
        for c in prof["q3_conditions"]:
            body = P["q3"]
            corr = "Yes" if c in ("neutral", "consistent") else "No"
            add("q3", c, "main", "marker", 0, 0, 0, with_caption(c, "marker", body), ["Yes", "No"], corr, None)
        if prof.get("q1"):
            for order in (0, 1):
                o = ["A", "B"] if order == 0 else ["B", "A"]
                if item["dataset"] == "synthetic":
                    text = P["q1_synth"].format(color=item["actor_color"], O1=o[0], O2=o[1])
                    add("q1", "none", "main", "marker", order, 0, 0, text, o, actor, None)
                elif item.get("lr_ok"):
                    text = P["q1_real"].format(O1=o[0], O2=o[1])
                    left_letter = actor if item["actor_side"] == "left" else distr
                    add("q1", "none", "main", "marker", order, 0, 0, text, o, left_letter, None)
    # ---- robustness: paraphrases (caption x question templates)
    if "robust" in blocks and item.get("robust"):
        for qt in range(len(P["q4"])):
            q4("none", "robust", q_tpl=qt)
            for ct in range(len(P["caption_paraphrases"])):
                q4("contra_2", "robust", cap_tpl=ct, q_tpl=qt)
    # ---- ablation: "answer from the image"
    if "ablation" in blocks and item.get("ablation"):
        q4("contra_2", "ablation", system=P["ablation_system"])
    # ---- left/right referencing scheme
    if "lr" in blocks and item.get("robust") and item.get("lr_ok"):
        for c in ("none", "neutral", "contra_2", "contra_4"):
            q4(c, "lr", scheme="lr")
    return out


def assign_subsets(items: list[dict], cfg, seed):
    rng = random.Random(seed + 7)
    by_ds: dict[str, list[dict]] = {}
    for it in items:
        by_ds.setdefault(it["dataset"], []).append(it)
    for ds, its in by_ds.items():
        ids = list(range(len(its)))
        rng.shuffle(ids)
        rn, an = cfg["profiles"]["robustness_n"], cfg["profiles"]["ablation_n"]
        for k, i in enumerate(ids):
            its[i]["robust"] = k < rn
            its[i]["ablation"] = k < an
    return items


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
