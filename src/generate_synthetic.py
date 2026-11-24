"""Procedurally drawn two-person scenes with exact ground truth.

Each scene: two flat-style figures with distinct shirt colours, one of whom (the actor) does one of four
actions with the only object in the scene. Randomised: actor side, which figure is "A", sizes, positions,
facing direction, background, skin/trouser colours, distractor pose (so pose alone is not a give-away
for "who is active": the distractor also moves).

Outputs: data/processed/synthetic/{items_synthetic.parquet, images/}
"""

from __future__ import annotations

import argparse
import math
import random

import pandas as pd
from PIL import Image, ImageDraw

from common import PROC, ROOT, area, config_hash, draw_markers, iou, load_config

OUT = PROC / "synthetic"

RGB = {
    "red": (214, 40, 40),
    "blue": (40, 90, 214),
    "green": (40, 160, 70),
    "yellow": (240, 200, 30),
    "purple": (130, 60, 180),
    "orange": (245, 130, 30),
}
SKIN = [(255, 220, 185), (234, 192, 134), (198, 134, 66), (141, 85, 36), (90, 56, 37)]
TROUSERS = [(40, 40, 60), (70, 70, 70), (90, 60, 30), (30, 50, 90), (110, 110, 120)]
BACKGROUNDS = [  # (sky, ground)
    ((170, 210, 240), (110, 170, 80)),
    ((200, 225, 245), (200, 180, 130)),
    ((180, 200, 220), (120, 120, 130)),
    ((235, 230, 220), (170, 140, 110)),
    ((150, 190, 230), (90, 140, 90)),
    ((220, 220, 230), (150, 160, 170)),
]


def limb_endpoint(p, length, ang_deg):
    """Point at `length` from p; angle 0 = straight down, +90 = pointing in +x."""
    a = math.radians(ang_deg)
    return (p[0] + length * math.sin(a), p[1] + length * math.cos(a))


class Figure:
    def __init__(self, cx, ground, h, shirt, skin, trousers, facing):
        self.cx, self.ground, self.h = cx, ground, h
        self.shirt, self.skin, self.trousers, self.facing = shirt, skin, trousers, facing
        self.head_r = h * 0.085
        self.leg = h * 0.46
        self.torso = h * 0.33
        self.arm = h * 0.36
        self.hip = (cx, ground - self.leg)
        self.neck = (cx, self.hip[1] - self.torso)
        self.head_c = (cx, self.neck[1] - self.head_r * 1.15)
        self.shoulder = (cx, self.neck[1] + h * 0.03)
        self.width = h * 0.12
        self.pts: list[tuple[float, float]] = []
        self.hand_front = self.hand_back = self.foot_front = None

    def draw(self, d: ImageDraw.ImageDraw, pose: dict):
        f = self.facing
        lw_leg = max(4, int(self.h * 0.055))
        lw_arm = max(3, int(self.h * 0.042))
        # legs: angles relative to down, mirrored by facing
        legs = []
        for key in ("leg_back", "leg_front"):
            ang = pose.get(key, 0) * f
            knee = limb_endpoint(self.hip, self.leg * 0.5, ang)
            foot = limb_endpoint(knee, self.leg * 0.5, ang + pose.get(key + "_knee", 0) * f)
            d.line([self.hip, knee, foot], fill=self.trousers, width=lw_leg, joint="curve")
            legs.append(foot)
        self.foot_front = legs[1]
        x0, x1 = self.cx - self.width / 2, self.cx + self.width / 2
        d.rounded_rectangle(
            [x0, self.neck[1], x1, self.hip[1] + self.h * 0.02],
            radius=int(self.width * 0.3),
            fill=self.shirt,
        )
        hands = []
        for key in ("arm_back", "arm_front"):
            ang = pose.get(key, 10) * f
            elbow = limb_endpoint(self.shoulder, self.arm * 0.5, ang)
            hand = limb_endpoint(elbow, self.arm * 0.5, ang + pose.get(key + "_elbow", 0) * f)
            d.line([self.shoulder, elbow, hand], fill=self.shirt, width=lw_arm, joint="curve")
            r = lw_arm * 0.6
            d.ellipse([hand[0] - r, hand[1] - r, hand[0] + r, hand[1] + r], fill=self.skin)
            hands.append(hand)
        self.hand_back, self.hand_front = hands
        hc, r = self.head_c, self.head_r
        d.ellipse(
            [hc[0] - r, hc[1] - r, hc[0] + r, hc[1] + r], fill=self.skin, outline=(60, 40, 30), width=1
        )
        ex = hc[0] + f * r * 0.45
        d.ellipse([ex - 2, hc[1] - 3, ex + 2, hc[1] + 1], fill=(30, 30, 30))
        self.pts = [self.head_c, (hc[0], hc[1] - r), *legs, *hands, (x0, self.neck[1]), (x1, self.hip[1])]

    def bounding_box(self, pad=0.03):
        xs = [p[0] for p in self.pts] + [self.head_c[0] - self.head_r, self.head_c[0] + self.head_r]
        ys = [p[1] for p in self.pts] + [self.head_c[1] - self.head_r]
        p = self.h * pad
        return [min(xs) - p, min(ys) - p, max(xs) + p, max(ys) + p]


ACTOR_POSES = {
    "kick_ball": {"leg_back": -8, "leg_front": 55, "leg_front_knee": 10, "arm_back": -40, "arm_front": 35},
    "hold_umbrella": {"arm_front": 150, "arm_front_elbow": 20, "arm_back": 12},
    "swing_racket": {
        "arm_front": 105,
        "arm_front_elbow": -15,
        "arm_back": -25,
        "leg_front": 18,
        "leg_back": -12,
    },
    "hold_flag": {"arm_front": 160, "arm_front_elbow": 5, "arm_back": 10},
}
DISTRACTOR_POSES = [
    {"arm_back": 10, "arm_front": 10},  # stand
    {"arm_front": 150, "arm_front_elbow": -40, "arm_back": 10},  # wave
    {"arm_back": 45, "arm_back_elbow": -110, "arm_front": 45, "arm_front_elbow": -110},  # hands on hips
    {"leg_back": -20, "leg_front": 20, "arm_back": -25, "arm_front": 25},  # walk
    {"arm_front": 90, "arm_back": 10},  # point
]


def draw_object(d, verb, fig: Figure, rng):
    """Draws the object attached to the actor; returns its bbox."""
    f = fig.facing
    s = fig.h
    if verb == "kick_ball":
        r = s * 0.065
        fx, fy = fig.foot_front
        c = (fx + f * r * 1.05, fy - r * 0.3)
        d.ellipse(
            [c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=(250, 250, 250), outline=(20, 20, 20), width=2
        )
        d.polygon(
            [
                (c[0], c[1] - r * 0.45),
                (c[0] + r * 0.43, c[1] - r * 0.14),
                (c[0] + r * 0.27, c[1] + r * 0.36),
                (c[0] - r * 0.27, c[1] + r * 0.36),
                (c[0] - r * 0.43, c[1] - r * 0.14),
            ],
            fill=(20, 20, 20),
        )
        return [c[0] - r, c[1] - r, c[0] + r, c[1] + r]
    if verb == "hold_umbrella":
        hx, hy = fig.hand_front
        top = (hx, fig.head_c[1] - s * 0.22)
        d.line([(hx, hy + s * 0.05), top], fill=(60, 60, 60), width=max(2, int(s * 0.015)))
        R = s * 0.26
        col = rng.choice([(30, 30, 30), (200, 30, 90), (30, 120, 200), (230, 180, 20)])
        d.pieslice(
            [top[0] - R, top[1] - R * 0.6, top[0] + R, top[1] + R * 0.6],
            180,
            360,
            fill=col,
            outline=(20, 20, 20),
        )
        d.arc(
            [hx - s * 0.03, hy + s * 0.02, hx + s * 0.03, hy + s * 0.08], 0, 180, fill=(60, 60, 60), width=2
        )
        return [top[0] - R, top[1] - R * 0.6, top[0] + R, hy + s * 0.08]
    if verb == "swing_racket":
        hx, hy = fig.hand_front
        ang = math.radians(rng.uniform(20, 60))
        L = s * 0.2
        end = (hx + f * L * math.cos(ang), hy - L * math.sin(ang))
        d.line([(hx, hy), end], fill=(40, 40, 40), width=max(3, int(s * 0.02)))
        rw, rh = s * 0.075, s * 0.1
        hc = (end[0] + f * rw * 0.7 * math.cos(ang), end[1] - rh * 0.7 * math.sin(ang))
        d.ellipse(
            [hc[0] - rw, hc[1] - rh, hc[0] + rw, hc[1] + rh],
            outline=(200, 30, 30),
            width=max(3, int(s * 0.018)),
        )
        for k in range(-2, 3):
            d.line(
                [(hc[0] + k * rw / 3, hc[1] - rh * 0.9), (hc[0] + k * rw / 3, hc[1] + rh * 0.9)],
                fill=(230, 230, 230),
                width=1,
            )
        xs = [hx, end[0], hc[0] - rw, hc[0] + rw]
        ys = [hy, end[1], hc[1] - rh, hc[1] + rh]
        return [min(xs), min(ys), max(xs), max(ys)]
    if verb == "hold_flag":
        hx, hy = fig.hand_front
        top = (hx, hy - s * 0.32)
        d.line([(hx, hy + s * 0.1), top], fill=(90, 70, 40), width=max(3, int(s * 0.018)))
        fw, fh = s * 0.22, s * 0.14
        col = rng.choice([(200, 30, 30), (30, 60, 180), (30, 150, 60), (250, 250, 250)])
        x2 = top[0] + f * fw
        d.polygon(
            [top, (x2, top[1] + fh * 0.1), (x2, top[1] + fh), (top[0], top[1] + fh * 0.9)],
            fill=col,
            outline=(30, 30, 30),
        )
        return [min(top[0], x2), top[1], max(top[0], x2), hy + s * 0.1]
    raise ValueError(verb)


def make_scene(cfg, k, verb, rng):
    W, H = cfg["synthetic"]["size"]
    sky, ground_col = rng.choice(BACKGROUNDS)
    im = Image.new("RGB", (W, H), sky)
    d = ImageDraw.Draw(im)
    horizon = rng.uniform(0.45, 0.6) * H
    d.rectangle([0, horizon, W, H], fill=ground_col)
    for _ in range(rng.randint(0, 3)):
        x, y, r = rng.uniform(0, W), rng.uniform(0, horizon * 0.6), rng.uniform(15, 40)
        d.ellipse([x - 2 * r, y - r, x + 2 * r, y + r], fill=(245, 245, 250))

    colors = rng.sample(cfg["synthetic"]["shirt_colors"], 2)
    actor_side = rng.choice(["left", "right"])
    xs = {"left": rng.uniform(0.2, 0.36) * W, "right": rng.uniform(0.64, 0.8) * W}
    figs = {}
    for role, side, col in (
        ("actor", actor_side, colors[0]),
        ("distractor", "right" if actor_side == "left" else "left", colors[1]),
    ):
        h = rng.uniform(0.5, 0.68) * H
        ground = rng.uniform(0.86, 0.95) * H
        facing = rng.choice([-1, 1])
        figs[role] = Figure(xs[side], ground, h, RGB[col], rng.choice(SKIN), rng.choice(TROUSERS), facing)
    figs["distractor"].draw(d, rng.choice(DISTRACTOR_POSES))
    figs["actor"].draw(d, ACTOR_POSES[verb])
    obj_box = draw_object(d, verb, figs["actor"], rng)

    ab, db = figs["actor"].bounding_box(), figs["distractor"].bounding_box()
    inside = lambda b: b[0] >= 2 and b[1] >= 2 and b[2] <= W - 2 and b[3] <= H - 2  # noqa: E731
    overlap = max(0, min(obj_box[2], db[2]) - max(obj_box[0], db[0])) * max(
        0, min(obj_box[3], db[3]) - max(obj_box[1], db[1])
    )
    if not (inside(ab) and inside(db) and inside(obj_box)) or overlap > 0 or iou(ab, db) > 0:
        return None
    actor_letter = rng.choice("AB")
    distr_letter = "B" if actor_letter == "A" else "A"
    item_id = f"synth_{k:04d}"
    img_dir = OUT / "images"
    mp, pp = img_dir / f"{item_id}_marked.jpg", img_dir / f"{item_id}_plain.jpg"
    draw_markers(im, {actor_letter: ab, distr_letter: db}).save(mp, quality=92)
    im.save(pp, quality=92)
    v = next(x for x in cfg["synthetic"]["verbs"] if x["key"] == verb)
    foil = rng.choice([x for x in cfg["synthetic"]["verbs"] if x["key"] != verb])
    ctr = lambda b: ((b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H)  # noqa: E731
    (ax, ay), (dx, dy), (ox, oy) = ctr(ab), ctr(db), ctr(obj_box)
    a_area, d_area = area(ab) / (W * H), area(db) / (W * H)
    return dict(
        item_id=item_id,
        dataset="synthetic",
        source="synthetic",
        source_file="",
        object=v["obj_np"][4:],
        verb=verb,
        verb_key=verb.replace("_", " "),
        ing=v["ing"],
        obj_np=v["obj_np"],
        q2=v["q2"],
        q2_foil=foil["q2"],
        actor_letter=actor_letter,
        actor_side=actor_side,
        lr_ok=True,
        actor_color=colors[0],
        distractor_color=colors[1],
        image_marked=mp.relative_to(ROOT).as_posix(),
        image_plain=pp.relative_to(ROOT).as_posix(),
        actor_box=ab,
        distractor_box=db,
        obj_box=obj_box,
        img_w=W,
        img_h=H,
        actor_area=a_area,
        distractor_area=d_area,
        min_person_area=min(a_area, d_area),
        person_iou=iou(ab, db),
        obj_area=area(obj_box) / (W * H),
        person_dist=((ax - dx) ** 2 + (ay - dy) ** 2) ** 0.5,
        obj_actor_dist=((ax - ox) ** 2 + (ay - oy) ** 2) ** 0.5,
        obj_distractor_dist=((dx - ox) ** 2 + (dy - oy) ** 2) ** 0.5,
        log_area_ratio=math.log(a_area / d_area),
        distractor_labels="",
        obj_detected=True,
        actor_facing_distractor=(figs["actor"].facing == (1 if actor_side == "left" else -1)),
        audit_status="exact",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-verb", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    n = a.n_per_verb or cfg["synthetic"]["n_per_verb"]
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["seed"] + 1)
    rows, k = [], 0
    for verb in [v["key"] for v in cfg["synthetic"]["verbs"]]:
        for _ in range(n):
            row = None
            while (
                row is None
            ):  # resample until the scene is valid (all in frame, object clear of distractor)
                row = make_scene(cfg, k, verb, rng)
            rows.append(row)
            k += 1
    df = pd.DataFrame(rows)
    df["person_area_threshold"] = None
    df["config_hash"] = config_hash()
    df.to_parquet(OUT / "items_synthetic.parquet", index=False)
    sheet = Image.new("RGB", (6 * 320, 4 * 240), "white")
    for i, r in enumerate(df.groupby("verb").head(6).itertuples()):
        t = Image.open(ROOT / r.image_marked).resize((320, 240))
        sheet.paste(t, ((i % 6) * 320, (i // 6) * 240))
    sheet.save(OUT / "contact_sheet.jpg", quality=88)
    print(df.groupby("verb").size(), "\n", df[["actor_side", "actor_letter"]].value_counts())


if __name__ == "__main__":
    main()
