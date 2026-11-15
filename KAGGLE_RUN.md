# Kaggle run: InternVL3-2B (step by step)

**Why Kaggle:** InternVL3-2B in fp16 needs more than the 6 GB RTX 4050 has, so Windows spills to system
RAM (3 s/pass, ~19 h). On a free Kaggle T4 (16 GB) it fits. The other models run on the laptop.

**Time:** ~10 min setup + roughly 1–3 h of GPU time for 952 items (≈22,800 forward passes). The exact
figure is printed after the smoke test (step 6). The free quota is ~30 GPU-hours a week, and one session can last up to 12 h.

---

## A. On your laptop (5 min)

1. Open a terminal in the repo folder (`ERB2/`) and pull the latest code:
   ```bash
   git pull
   ```
2. Build the bundle (code + config + the 952 items and their images, ~60 MB; **no keys** are included):
   ```bash
   .venv/Scripts/python scripts/make_kaggle_bundle.py
   ```
   It prints `...\ERB2\erb2_bundle.zip (N files, ~61 MB)`.

## B. Upload the bundle as a Kaggle dataset (3 min)

3. Go to <https://www.kaggle.com> → sign in → left sidebar **Datasets** → **+ New Dataset**.
   - Drag in `erb2_bundle.zip`.
   - Title: `erb2-bundle`. Visibility: **Private**.
   - Click **Create** and wait until processing finishes (Kaggle may unzip it; either way is fine).

## C. Create the notebook (3 min)

4. Left sidebar **Code** → **+ New Notebook**. Then:
   - **File → Import Notebook** → upload `notebooks/kaggle_runner.ipynb` from the repo.
   - Right panel **Session options** (or *Settings*):
     - **Accelerator → GPU T4 x2** (one GPU is used; P100 also works).
     - **Internet → On.** This needs a phone-verified Kaggle account; the model weights download from Hugging Face.
     - **Persistence:** leave the default.
   - Right panel **Input → + Add Input** → **Your Datasets** → `erb2-bundle` → **Add**.

## D. Run (watch the first 15 min, then leave it)

5. Run the cells **one at a time** the first time:
   1. *Markdown* (instructions). Nothing to run.
   2. **Find bundle.** It must print `bundle from /kaggle/input/...` and a list containing `src`, `configs`, `data`.
      If it says *bundle not found*, the dataset isn't attached (step 4).
   3. **pip install.** About 1–2 min. The last line prints the torch version and `Tesla T4`.
6. **Smoke test** (20 items per dataset, ~2–5 min including the 4.4 GB model download). Check that the output has
   - `"n_errors": 0`
   - a line like `~X s/item -> 1,200 items ~ Y h`. Your full run ≈ **Y × 0.8 h** (952 items). If Y × 0.8 is over ~11 h, stop and tell us.
7. **Full run.** Run the `for m in MODELS` cell. It shows a progress bar, `internvl3-2b: 952 items, 22800 queries`.
   You can close the browser tab: use **Save Version → Save & Run All (Commit)** instead of interactive mode if you
   want it to run unattended (the commit runs every cell top to bottom; the smoke-test cell just adds a few minutes).
8. **Zip results.** The last cell writes `/kaggle/working/erb2_results.zip` and lists `results/`:
   you should see `internvl3-2b.jsonl`, `internvl3-2b.parquet`, `internvl3-2b.meta.json`.

## E. Bring results back (2 min)

9. Download `erb2_results.zip`:
   - interactive session: right panel **Output** → `/kaggle/working/erb2_results.zip` → ⋮ → **Download**;
   - committed version: notebook page → **Output** tab → download.
10. Unzip into the repo root so the files land in `ERB2/results/` (the zip already contains the `results/` folder):
    ```bash
    cd ERB2 && unzip -o erb2_results.zip
    ```
11. Commit them:
    ```bash
    git add results/internvl3-2b.* && git commit -m "InternVL3-2B full run (Kaggle T4)"
    ```
    Then rerun the analysis (`python src/05_analysis.py`) or tell Claude it's in.

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| `bundle not found under /kaggle/input` | The dataset isn't attached: **Input → + Add Input** |
| `OSError ... huggingface.co` / connection error | Internet is off, or the account isn't phone-verified |
| `CUDA out of memory` | Accelerator isn't a GPU, or another notebook holds it. Restart the session |
| Session hit 12 h / got disconnected | Download the partial `results/internvl3-2b.jsonl`, add it to the bundle (`make_kaggle_bundle.py` includes `results/*.jsonl`), re-upload as a **new version** of the dataset, run again. It **resumes** and skips finished queries |
| Smoke test shows errors | Copy the first `error:` line and send it to us |

**Do not** edit `configs/verbs.yaml` on Kaggle. Prompts and settings must match the laptop runs (each results meta file records the `config_hash`, so a mismatch would show up in the analysis).
