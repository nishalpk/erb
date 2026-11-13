# ERB2: Entity–Role Binding Under Contradictory Text

Do vision-language models keep track of **who** is doing an action when a caption says otherwise? Two people
are marked **A** and **B**, exactly one of them does the action, and a caption falsely names the other one at
four confidence levels. Nothing is trained. See `PLAN.md` for the design and `PREREGISTRATION.md` for the
frozen hypotheses, definitions and deviations.

## Pipeline

| Step | Command | Output |
|---|---|---|
| D0 schema | `python src/00_inspect_schema.py` | `docs/schema_notes.md` |
| D1 real subset | `python src/01_build_real.py` | `data/processed/real/{items_real.parquet, funnel.csv}` |
| D2 audit | `python src/02_audit_sample.py prepare` → label in browser → `... score` | `reports/audit/`, `items_real_audited.parquet` |
| D4 synthetic | `python src/03_make_synthetic.py` | `data/processed/synthetic/` (800 scenes) |
| D3 smoke test | `python src/04_run_models.py --model llava-ov-0.5b --limit 50 --tag smoke` | timing per pass |
| D5 full runs | `python src/04_run_models.py --model <name>` (resumable) | `results/<model>.{jsonl,parquet,meta.json}` |
| D6 analysis | `python src/05_analysis.py` (or `src/05_analysis.ipynb`) | `reports/results.md`, `reports/figures/` |
| D7 stretch | `python src/06_probe_patch.py --model qwen2.5-vl-3b` | `fig8_probe_patch.png`, H5 verdict |
| Pipeline check | `python src/04_run_models.py --model mock` then `python src/05_analysis.py --models mock` | simulated model with known effects |

Model names are defined in `configs/verbs.yaml` → `models`. Everything (whitelists, thresholds, caption
templates, query profiles) lives in that one file, and its hash is stored with every result.

## Hardware without the RTX 5080

* **This laptop (RTX 4050, 6 GB):** `llava-ov-0.5b`, `smolvlm2-2.2b`, `internvl3-2b` in fp16; `qwen2.5-vl-3b` in 4-bit (set in the config).
* **Kaggle free T4 x2 (recommended for the full runs):** `python scripts/make_kaggle_bundle.py`, upload the zip
  as a dataset, run `notebooks/kaggle_runner.ipynb`. Qwen-3B runs unquantised there.
* **Keys** live in the git-ignored `.env` (`GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN`), loaded by `src/common.py`.
* **Gemini API (closed-model reference):** `python src/04_run_models.py --model gemini-3.5-flash-lite --profile core --blocks main --limit 75`
  (75 real + 75 synthetic, ~14 calls/item). When the daily quota runs out the run stops cleanly; rerun the same
  command later to resume. As of 2026-11-13 an AI Studio free key: `gemini-2.5-flash-lite` is closed to new users,
  `gemini-2.5-flash` allows ~20 requests/day, and no model returns logprobs, so the readout is the parsed letter.
  Report it separately as a reference point.
* **OpenRouter:** model `openrouter-gemma4` (`google/gemma-4-31b-it:free`); the free tier allows 50 requests/day, a token reference only.

## What changed versus the plan (and why)

* **Annotation source.** The HF parquet `zhimeng/hico_det` keeps one box pair per HOI category and drops
  the second person in multi-person HOIs, so CVC1/3/4 can't be evaluated from it. Real items come from
  **V-COCO** (COCO boxes every person; every V-COCO person is labelled for every action) and **HICO-DET
  train** (full HAKE instance-level release).
* **Real-subset size.** The plan's five-verb whitelist gives **0** strict items on HICO train, because the
  other person almost always has some label for the object ("block ball", "catch frisbee"). The whitelists were widened to
  exclusive-use actions, and each entry lists the distractor labels that cannot make the false caption true
  (`allowed_other`). Even so, the real subset is a few hundred items at most (see `funnel.csv`), below the
  target of 400. The **800 synthetic scenes** carry the statistical power, as PLAN 2b allows.
* **CVC1 for HICO** also uses the detector, because HICO-DET doesn't box bystanders.
* **Patching pair** = `consistent` → `contra_2` (same token length), not `none` → contra.

## Layout

```
configs/verbs.yaml        whitelists, thresholds, prompt templates, query profiles, model registry
src/common.py             box maths, marker overlay, query construction (conditions x questions x orders)
src/backends.py           hf_local (logit readout) | gemini_api | openrouter | mock
src/analysis.py           Wilson, McNemar, cluster bootstrap, Cochran-Armitage, GEE, taxonomy, robustness
src/0*_*.py               pipeline steps (table above)
reports/                  results.md, figures, tables, audit pages, rebinding gallery
```
