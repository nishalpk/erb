"""PLACEHOLDER audit labels for annotators R2 and R3 (simulated, NOT real human labels).

Lets `src/02_audit_sample.py score` and `05_analysis.py --audited-only` run end to end before the real
R2/R3 labelling is done. Every row is tagged in `notes`, and `score` prints a warning banner while any
placeholder row is present. Replace with the real exports (Export CSV in reports/audit/audit_R*.html).

    python scripts/make_placeholder_audit.py            # writes reports/audit/labels_R2.csv, labels_R3.csv
    python scripts/make_placeholder_audit.py --remove   # deletes them again (only if still placeholders)

Simulation: on the shared items each field copies R1 and is resampled from R1's marginal with p=0.08;
on an annotator's own items fields are drawn from R1's marginals, with `who` = the annotated actor w.p. 0.95.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "reports" / "audit"
FIELDS = ["exactly_one", "who", "other_could", "second_object"]
TAG = "PLACEHOLDER: simulated label, replace with real annotation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", default="R2,R3")
    ap.add_argument("--flip", type=float, default=0.08)
    ap.add_argument("--remove", action="store_true")
    a = ap.parse_args()
    anns = a.annotators.split(",")
    if a.remove:
        for r in anns:
            p = AUD / f"labels_{r}.csv"
            if p.exists() and (pd.read_csv(p, dtype=str).notes.fillna("").str.startswith("PLACEHOLDER")).all():
                p.unlink()
                print("removed", p)
        return
    rng = np.random.default_rng(2026)
    r1 = pd.read_csv(AUD / "labels_R1.csv", dtype=str).fillna("").set_index("item_id")
    marg = {f: r1[f].value_counts(normalize=True) for f in FIELDS}
    items = pd.read_parquet(ROOT / "data" / "processed" / "real" / "items_real.parquet").set_index("item_id")
    assign = pd.read_csv(AUD / "assignment.csv")
    for r in anns:
        out = AUD / f"labels_{r}.csv"
        if out.exists() and not (pd.read_csv(out, dtype=str).notes.fillna("").str.startswith("PLACEHOLDER")).all():
            sys.exit(f"{out} holds real labels; not overwriting")
        rows = []
        for it in assign[assign.annotator == r].item_id:
            row = {"item_id": it}
            for f in FIELDS:
                draw = lambda: rng.choice(marg[f].index, p=marg[f].values)  # noqa: E731
                if it in r1.index and r1.loc[it, f]:
                    row[f] = draw() if rng.random() < a.flip else r1.loc[it, f]
                elif f == "who":
                    actor = items.loc[it, "actor_letter"]
                    row[f] = actor if rng.random() < 0.95 else ("B" if actor == "A" else "A")
                else:
                    row[f] = draw()
            row["notes"] = TAG
            rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"wrote {out} ({len(rows)} rows, PLACEHOLDER)")


if __name__ == "__main__":
    main()
