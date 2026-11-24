"""Shared helpers: paths, config, box maths, marker rendering, query construction."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path=None):
    """Read KEY=VALUE lines from the git-ignored .env into os.environ (existing variables win)."""
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


def download_hake_annotations(cfg):
    from huggingface_hub import hf_hub_download

    real = cfg["real"]
    download_dir = RAW / "hake"
    annotation_file = hf_hub_download(
        real["hake_repo"], real["hake_file"], repo_type="dataset", local_dir=download_dir
    )
    hoi_list_file = hf_hub_download(
        real["hake_repo"], real["hoi_list_file"], repo_type="dataset", local_dir=download_dir
    )
    return Path(annotation_file), Path(hoi_list_file)


def load_hoi_list(path) -> dict[int, tuple[str, str]]:
    interactions = {}
    for line in Path(path).read_text().splitlines():
        columns = line.split()
        if len(columns) >= 3 and columns[0].isdigit():
            interactions[int(columns[0])] = (columns[1], columns[2])
    return interactions


# Boxes use (x1, y1, x2, y2) coordinates.


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = area(a) + area(b) - inter
    return inter / u if u > 0 else 0.0


def cluster_boxes(boxes, thr):
    """Group boxes greedily by IoU and return cluster labels and mean boxes."""
    labels = []
    clusters = []
    for box in boxes:
        best_cluster, best_iou = -1, thr
        for index, members in enumerate(clusters):
            overlap = max(iou(box, member) for member in members)
            if overlap >= best_iou:
                best_cluster, best_iou = index, overlap
        if best_cluster == -1:
            best_cluster = len(clusters)
            clusters.append([])
        clusters[best_cluster].append(box)
        labels.append(best_cluster)

    mean_boxes = [
        list(members[0])
        if len(members) == 1
        else [sum(coordinate) / len(members) for coordinate in zip(*members)]
        for members in clusters
    ]
    return labels, mean_boxes



def _marker_font(size):
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
    font = _marker_font(fs)
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
        d.rectangle(
            [tx, ty, tx + tw + 2 * pad, ty + fs + 2 * pad], fill=(255, 221, 0), outline=(0, 0, 0), width=2
        )
        d.text((tx + pad, ty + pad * 0.6), letter, fill=(0, 0, 0), font=font)
    return im


def resize_to_fit(img: Image.Image, max_side: int) -> Image.Image:
    width, height = img.size
    scale = max_side / max(width, height)
    if scale >= 1:
        return img
    return img.resize((round(width * scale), round(height * scale)), Image.LANCZOS)


def load_model_image(path: Path, max_side: int) -> Image.Image:
    with Image.open(path) as image:
        return resize_to_fit(image.convert("RGB"), max_side)



@dataclass
class Query:
    qid: str
    item_id: str
    dataset: str
    question: str  # q1 q2 q2_foil q3 q4
    condition: str  # none neutral consistent contra_1..4
    block: str  # main | robust | ablation | lr
    scheme: str  # marker | lr
    order: int  # 0 = options as (O1,O2)=(A,B)/(left,right); 1 = reversed
    cap_tpl: int
    q_tpl: int
    image_path: str
    system: str | None
    prompt: str
    options: list[str]
    correct: str
    distractor: str | None


def _entity_reference(scheme, actor_letter, actor_side, which):
    """Entity reference string for 'actor' or 'distractor'."""
    if scheme == "marker":
        other = "B" if actor_letter == "A" else "A"
        return actor_letter if which == "actor" else other
    other = "right" if actor_side == "left" else "left"
    side = actor_side if which == "actor" else other
    return f"the person on the {side}"


def _capitalize_first(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def make_caption(cfg, cond, item, scheme, tpl=0):
    if cond == "none":
        return None
    prompts = cfg["prompts"]
    if cond == "neutral":
        first = _entity_reference(scheme, "A", "left", "actor")
        second = _entity_reference(scheme, "A", "left", "distractor")
        return prompts["captions"]["neutral"].format(X1=first, X2=second)
    who = "actor" if cond == "consistent" else "distractor"
    person = _entity_reference(scheme, item["actor_letter"], item["actor_side"], who)
    template = prompts["captions"][cond]
    if tpl and cond == "contra_2":
        template = prompts["caption_paraphrases"][tpl]
    caption = template.format(X=person, ing=item["ing"], obj=item["obj_np"])
    return _capitalize_first(caption.replace("  ", " ")).replace(". the", ". The")


def build_queries(cfg, item: dict, profile: str, blocks: set[str]) -> list[Query]:
    prompts = cfg["prompts"]
    profile_config = cfg["profiles"][profile]
    queries: list[Query] = []
    actor = item["actor_letter"]
    other_letter = "B" if actor == "A" else "A"

    def add_query(
        question,
        cond,
        prompt,
        options,
        correct,
        *,
        block="main",
        scheme="marker",
        order=0,
        cap_tpl=0,
        q_tpl=0,
        distractor=None,
        system=None,
        image_key="image_marked",
    ):
        qid = f"{item['item_id']}|{question}|{cond}|{block}|{scheme}|o{order}|c{cap_tpl}|q{q_tpl}"
        queries.append(
            Query(
                qid=qid,
                item_id=item["item_id"],
                dataset=item["dataset"],
                question=question,
                condition=cond,
                block=block,
                scheme=scheme,
                order=order,
                cap_tpl=cap_tpl,
                q_tpl=q_tpl,
                image_path=item[image_key],
                system=system,
                prompt=prompt,
                options=options,
                correct=correct,
                distractor=distractor,
            )
        )

    def with_caption(cond, scheme, body, cap_tpl=0):
        caption = make_caption(cfg, cond, item, scheme, cap_tpl)
        if caption is None:
            return body
        prefix = prompts["caption_wrapper"].format(caption=caption)
        return f"{prefix}\n{body}"

    def add_binding_queries(cond, block, scheme="marker", cap_tpl=0, q_tpl=0, system=None):
        if scheme == "marker":
            base_options = ["A", "B"]
            correct, distractor = actor, other_letter
            image_key = "image_marked"
        else:
            base_options = ["left", "right"]
            correct = item["actor_side"]
            distractor = "right" if correct == "left" else "left"
            image_key = "image_plain"
        suffix = prompts[f"q4_suffix_{scheme}"]
        for order, options in enumerate((base_options, base_options[::-1])):
            first, second = options
            if scheme == "lr":
                first, second = (f"the person on the {option}" for option in options)
            question = prompts["q4"][q_tpl].format(ing=item["ing"], obj=item["obj_np"], O1=first, O2=second)
            body = question + " " + suffix.format(O1=options[0], O2=options[1])
            add_query(
                "q4",
                cond,
                with_caption(cond, scheme, body, cap_tpl),
                options,
                correct,
                block=block,
                scheme=scheme,
                order=order,
                cap_tpl=cap_tpl,
                q_tpl=q_tpl,
                distractor=distractor,
                system=system,
                image_key=image_key,
            )

    if "main" in blocks:
        for condition in profile_config["q4_conditions"]:
            add_binding_queries(condition, "main")
        for condition in profile_config["q2_conditions"]:
            body = prompts["q2"].format(q2=item["q2"])
            add_query("q2", condition, with_caption(condition, "marker", body), ["Yes", "No"], "Yes")
        if profile_config.get("q2_foil"):
            body = prompts["q2"].format(q2=item["q2_foil"])
            add_query("q2_foil", "none", body, ["Yes", "No"], "No")
        for condition in profile_config["q3_conditions"]:
            correct = "Yes" if condition in ("neutral", "consistent") else "No"
            body = with_caption(condition, "marker", prompts["q3"])
            add_query("q3", condition, body, ["Yes", "No"], correct)
        if profile_config.get("q1"):
            for order, options in enumerate((["A", "B"], ["B", "A"])):
                first, second = options
                if item["dataset"] == "synthetic":
                    body = prompts["q1_synth"].format(color=item["actor_color"], O1=first, O2=second)
                    add_query("q1", "none", body, options, actor, order=order)
                elif item.get("lr_ok"):
                    body = prompts["q1_real"].format(O1=first, O2=second)
                    left_letter = actor if item["actor_side"] == "left" else other_letter
                    add_query("q1", "none", body, options, left_letter, order=order)

    if "robust" in blocks and item.get("robust"):
        for question_index in range(len(prompts["q4"])):
            add_binding_queries("none", "robust", q_tpl=question_index)
            for caption_index in range(len(prompts["caption_paraphrases"])):
                add_binding_queries("contra_2", "robust", cap_tpl=caption_index, q_tpl=question_index)
    if "ablation" in blocks and item.get("ablation"):
        add_binding_queries("contra_2", "ablation", system=prompts["ablation_system"])
    if "lr" in blocks and item.get("robust") and item.get("lr_ok"):
        for condition in ("none", "neutral", "contra_2", "contra_4"):
            add_binding_queries(condition, "lr", scheme="lr")
    return queries


def assign_subsets(items: list[dict], cfg, seed):
    rng = random.Random(seed + 7)
    by_dataset: dict[str, list[dict]] = {}
    for item in items:
        by_dataset.setdefault(item["dataset"], []).append(item)
    robustness_count = cfg["profiles"]["robustness_n"]
    ablation_count = cfg["profiles"]["ablation_n"]
    for dataset_items in by_dataset.values():
        rng.shuffle(dataset_items)
        for rank, item in enumerate(dataset_items):
            item["robust"] = rank < robustness_count
            item["ablation"] = rank < ablation_count
    return items
