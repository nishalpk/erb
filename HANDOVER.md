# Handover: state on 2026-11-13 (evening)

Read `README.md` (pipeline) and `PREREGISTRATION.md` (hypotheses + every change from `PLAN.md`) first.
This file is what's **done** and what's **left**, in order.

## 1. Set up after cloning

```bash
python -m venv .venv
# CUDA GPU: install torch first from the PyTorch index (cu128 works for RTX 40/50 series)
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt     # exact versions used: requirements-lock.txt
cp .env.example .env                                         # fill in your own keys (never commit .env)
```
On Windows, set `PYTHONIOENCODING=utf-8` in your shell. Model weights download on first use (~15 GB for the four local models).

Check that everything runs (no GPU or keys needed, ~1 min):
```bash
.venv/Scripts/python src/04_run_models.py --model mock --datasets synthetic
.venv/Scripts/python src/05_analysis.py --models mock --boot 200
```
`reports/results.md` and `reports/figures/` in the repo currently come from this **mock** model, so they are pipeline checks, not findings.

## 2. Done

| Step | State |
|---|---|
| D0 schema | `docs/schema_notes.md`. The HF `zhimeng/hico_det` copy drops the second person per action, so the sources are HAKE HICO-DET train + V-COCO |
| D1 real subset | **152 items** (V-COCO 138, HICO 14) in `data/processed/real/` (committed, images included). Funnel: `funnel.csv`. Below the 400 target; see PREREGISTRATION deviations |
| D4 synthetic | 800 scenes in `data/processed/synthetic/` (committed) |
| D2 audit tooling | `reports/audit/audit_R{1,2,3}.html` generated (100 shared items + the rest split). **Not labelled yet** |
| D3 smoke tests | All 4 local models + Gemini + OpenRouter verified (see timings below) |
| D6 analysis, D7 probe | Code done; analysis checked on the mock model; probe **not run** |
| Kaggle | `notebooks/kaggle_runner.ipynb`; bundle via `python scripts/make_kaggle_bundle.py` |

Measured on an RTX 4050 6 GB (time per forward pass, all 952 items ≈ 22,800 passes per model):

| Model | s/pass | full run | note |
|---|---|---|---|
| smolvlm2-2.2b | 0.13 | ~0.8 h | |
| llava-ov-0.5b | 0.34 | ~2.2 h | strong "A" bias; order averaging handles it |
| qwen2.5-vl-3b (4-bit) | 0.49 | ~3.1 h | unquantised on Kaggle T4 (the notebook sets this) |
| internvl3-2b | 3.0 | ~19 h | fp16 spills past 6 GB → run on Kaggle T4 |

## 3. TODO (in this order)

1. **Full model runs (D5).** Partial results are committed and runs **resume** from `results/<model>.jsonl`:
   - `smolvlm2-2.2b`: ~3,900 / 22,800 queries done
   - `llava-ov-0.5b`, `qwen2.5-vl-3b`, `internvl3-2b`: not started
   ```bash
   .venv/Scripts/python src/04_run_models.py --model smolvlm2-2.2b
   .venv/Scripts/python src/04_run_models.py --model llava-ov-0.5b
   .venv/Scripts/python src/04_run_models.py --model qwen2.5-vl-3b
   ```
   InternVL3-2B (and Qwen-7B, stretch): Kaggle T4 via `notebooks/kaggle_runner.ipynb`, or locally with `quant: 4bit` in `configs/verbs.yaml` (say so in the report if you do).
2. **Gemini reference.** `gemini-3.5-flash-lite`: 401 / 2,100 calls done. The free tier allows ~500 calls/day; rerun daily until it finishes, or pay a few dollars:
   ```bash
   .venv/Scripts/python src/04_run_models.py --model gemini-3.5-flash-lite --profile core --blocks main --limit 75
   ```
   It reads answers as text (no logprobs on AI Studio keys). Keep it out of the pooled regression.
3. **Manual audit (CVC5). Needs 3 people.** Serve the pages and open one per person:
   ```bash
   .venv/Scripts/python -m http.server 8765 --directory reports
   ```
   `http://localhost:8765/audit/audit_R1.html` (R2, R3). Labels save in the browser as you go; click **Export CSV**, put `labels_R*.csv` into `reports/audit/`, then:
   ```bash
   .venv/Scripts/python src/02_audit_sample.py score
   .venv/Scripts/python src/02_audit_sample.py triage     # after model runs: items all models miss -> re-review
   ```
4. **Analysis (D6):** `python src/05_analysis.py`, then `python src/05_analysis.py --audited-only`. Fill the hypothesis table from `reports/results.md`.
5. **Probe/patching (D7):** `python src/06_probe_patch.py --model qwen2.5-vl-3b --n 200` → `fig8`, H5 verdict.
6. **Deck (D8):** outline in PLAN.md section 10; figures in `reports/figures/`.

## 4. Future work (deferred)
- More real items: HICO-DET test split (needs a full multi-pair annotation copy), other HOI sets, or allowing tiny background bystanders.
- More API models once the budget allows (OpenRouter free tier = 50 requests/day; `gemini-2.5-flash` ≈ 20/day).
- 7B models, activation patching across models, human baseline (PLAN section 9).

## 5. Gotchas
- Keys come from `.env`. The keys used on 2026-11-13 were pasted into a chat, so they should be **rotated**; use your own.
- `data/raw/` is not committed; `src/00`/`src/01` re-download it (~500 MB COCO annotations + candidate images). You only need to if you change the filters; the processed items are committed.
- Changing `configs/verbs.yaml` changes `config_hash` in new results; don't change prompts in the middle of a set of runs.
