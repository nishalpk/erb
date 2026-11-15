"""Generates src/05_analysis.ipynb and notebooks/kaggle_runner.ipynb (kept as code so they stay in sync)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


# ------------------------------------------------------------------ analysis notebook
nb = nbf.v4.new_notebook()
nb.cells = [
    md("""
# 05 · Analysis: Entity–Role Binding under contradictory text
Interactive companion to `src/05_analysis.py` (which writes `reports/results.md` and all figures).
Definitions and decision rules: `PREREGISTRATION.md`. Set `MODELS = ["mock"]` to see the pipeline on the simulated model.
"""),
    code("""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd() if (Path.cwd() / "analysis.py").exists() else Path.cwd() / "src"))
import pandas as pd, numpy as np
import analysis as A
from IPython.display import Image, display, Markdown
pd.set_option("display.width", 200, "display.max_columns", 30)

MODELS = None           # e.g. ["smolvlm2-2.2b", "internvl3-2b"]; None = every non-mock result
AUDITED_ONLY = False
items = A.load_items()
res = A.load_results(MODELS)
if AUDITED_ONLY:
    res = res[res.item_id.isin(items[items.audit_keep == True].item_id)]
res = res[res.item_id.isin(items.item_id)]
res.groupby(["model", "dataset"]).item_id.nunique()
"""),
    md("## Competence set S_m and reassignment rates"),
    code("""
q4 = A.q4_table(res)
S = A.competence(res, q4)
display(S.groupby("model").size().rename("|S_m|"))
rr = A.rr_table(q4, S)
rr.pivot_table(index=["model", "dataset"], columns="condition", values="RR", observed=True).round(3)
"""),
    md("## Capability panel and letter bias"),
    code("""
display(A.capability(res, q4).pivot_table(index=["model", "dataset"], columns="task", values="acc").round(3))
A.letter_bias(res)
"""),
    md("## H1 · contradictory caption vs neutral / none (McNemar, cluster bootstrap)"),
    code("""
h1 = A.h1(q4, S, n_boot=2000)
h1[h1.test == "contra_2"].round(4)
"""),
    md("## H2 · dose-response"),
    code("""
trend, gee = A.h2(q4, S, items)
display(trend)
for ds, g in gee.items():
    print(ds, {k: v for k, v in g.items() if k != "summary"})
"""),
    md("## H3 / H4 · dissociation and awareness"),
    code("""
tax = A.taxonomy(res, q4, S)
display(A.h3_h4(tax).round(3))
tax.groupby(["model", "dataset"]).outcome.value_counts(normalize=True).unstack().round(3)
"""),
    md("## Robustness and error analysis"),
    code("""
rob = A.robustness(res, S)
for k, v in rob.items():
    display(Markdown(f"**{k}**")); display(v.round(3) if hasattr(v, "round") else v)
A.difficulty_regression(q4, S, items)
"""),
    md("## Figures (run `python src/05_analysis.py` first)"),
    code("""
for f in sorted((Path("..") if Path.cwd().name == "src" else Path(".")).joinpath("reports/figures").glob("fig*.png")):
    display(Markdown(f"**{f.name}**")); display(Image(filename=str(f)))
"""),
]
nbf.write(nb, ROOT / "src" / "05_analysis.ipynb")

# ------------------------------------------------------------------ Kaggle runner
kb = nbf.v4.new_notebook()
kb.cells = [
    md("""
# ERB · Kaggle runner (free T4 x2)
1. Locally: `python scripts/make_kaggle_bundle.py` → `erb2_bundle.zip`.
2. Kaggle → *Datasets → New dataset* → upload the zip (name it `erb2-bundle`).
3. New notebook → *Add input* → that dataset; *Settings → Accelerator → GPU T4 x2*; Internet **on**.
4. Run all. Download `/kaggle/working/erb2_results.zip` and unzip into the repo's `results/`.

T4 has no bf16: models load in fp16 with SDPA attention. The run resumes from `results/*.jsonl`, so a
session that hits the 12 h limit can be continued by re-running with the previous results attached.
"""),
    code("""
import os, glob, shutil, subprocess, zipfile
# Kaggle mounts datasets at varying depths (e.g. /kaggle/input/datasets/<user>/<name>/), and may
# auto-extract the zip. Walk the whole input tree for either the zip or an extracted folder with src/.
WORK = "/kaggle/working/erb2"
zip_path, folder = None, None
for root, dirs, files in os.walk("/kaggle/input"):
    if "erb2_bundle.zip" in files:
        zip_path = os.path.join(root, "erb2_bundle.zip"); break
    if "src" in dirs and os.path.exists(os.path.join(root, "src", "04_run_models.py")):
        folder = root; break
if zip_path:
    zipfile.ZipFile(zip_path).extractall(WORK)
elif folder:
    shutil.copytree(folder, WORK, dirs_exist_ok=True)
else:
    raise SystemExit("bundle not found under /kaggle/input - check that the dataset is attached")
os.chdir(WORK); print("bundle from", zip_path or folder); print(os.listdir())
"""),
    code("""
!pip install -q -U "transformers>=4.57" accelerate bitsandbytes num2words pyyaml pyarrow
import torch; print(torch.__version__, torch.cuda.get_device_name(0))
"""),
    code("""
# Only the model that does not fit the 6 GB laptop. The others run locally (Qwen-3B there in 4-bit).
MODELS = ["internvl3-2b"]   # optional stretch: + "qwen2.5-vl-7b" (4-bit)
"""),
    code("""
# Smoke test + timing (20 items per dataset, ~1-2 min). Check: n_errors 0, note the s/item line.
!python src/04_run_models.py --model internvl3-2b --limit 20 --blocks main --tag smoke
"""),
    code("""
for m in MODELS:
    print("=" * 30, m)
    subprocess.run(["python", "src/04_run_models.py", "--model", m], check=False)
"""),
    code("""
with zipfile.ZipFile("/kaggle/working/erb2_results.zip", "w") as z:
    for f in glob.glob("results/*"):
        if "smoke" not in f:
            z.write(f)
print(os.listdir("results"))
"""),
]
(ROOT / "notebooks").mkdir(exist_ok=True)
nbf.write(kb, ROOT / "notebooks" / "kaggle_runner.ipynb")
print("wrote src/05_analysis.ipynb and notebooks/kaggle_runner.ipynb")
