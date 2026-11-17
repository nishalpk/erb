# Agent handover: ERB2 (Entity–Role Binding under contradictory text)

This is for an AI coding agent (Claude Code, Cursor, Codex, etc.) picking up the **whole project**.
Read it before you touch anything. Human-oriented notes are in `HANDOVER.md`. Running InternVL on Kaggle is covered in `KAGGLE_RUN.md`.
State as of **2026-11-15, 22:15**.

---

## 1. What the project is (one paragraph)

The question is whether small open vision-language models (VLMs) **re-bind an action to the wrong person** when a caption contradicts the image, or whether they just stop perceiving it.

Each item is an image with exactly two people, marked **A** and **B**. Person A does an action, for example riding a horse, and B is a bystander. We show the image with a caption. The caption is either absent, neutral, or consistent with the image, or it is one of four contradiction levels (`contra_1..4`) that claim **B** does the action, each more confidently than the last. Then we ask, each question in a fresh context:

| Question | Asks | Answer type |
|---|---|---|
| Q1 | Identity | yes/no |
| Q2 | Is the action present? | yes/no, plus a foil control |
| Q3 | Does the caption conflict with the image? | yes/no |
| Q4 | **Who** is doing the action? | A or B (the binding question) |

Answers are read from **logits** (forced choice, both option orders, averaged), not from generated text. Pre-registered hypotheses H1–H5 are in `PREREGISTRATION.md`:

- **H1:** a contradiction flips Q4.
- **H2:** the effect grows with the confidence level (dose-response).
- **H3:** the model rebinds the action rather than losing perception: Q2 stays correct while Q4 flips.
- **H4:** the model notices the conflict (Q3) yet still follows the caption.
- **H5:** a linear probe or activation patching shows the visual binding is still present internally.

## 2. Repository map

| Path | Role |
|---|---|
| `PLAN.md` | Original spec (sections 0–11). Section 10 is the deck outline |
| `PREREGISTRATION.md` | Hypotheses, definitions, decision rules, **Deviations** list. Append to it; never rewrite earlier entries |
| `configs/verbs.yaml` | **Single source of truth**: verbs, whitelists, thresholds, prompts, models, runtime caps. Its hash is stored in every result. **Do not edit prompts or conditions mid-run** |
| `src/common.py` | Paths, `.env` loading, `build_queries` (the condition × question × order grid), marker drawing |
| `src/00_inspect_schema.py` | D0: dataset schema notes → `docs/schema_notes.md` |
| `src/01_build_real.py` | D1: HAKE HICO-DET + V-COCO → CVC1–4 filter funnel → `data/processed/real/` (152 items) |
| `src/02_audit_sample.py` | D2: `prepare` (HTML audit pages), `score` (Fleiss κ, precision, CVC5), `triage` |
| `src/03_make_synthetic.py` | D4: 800 procedurally drawn two-person scenes |
| `src/backends.py` | `HFLocal` (logit readout), `GeminiAPI` (text readout), `OpenRouter`, `Mock` |
| `src/04_run_models.py` | D5: runs a model; resumable JSONL → parquet + meta.json |
| `src/analysis.py` | All statistics (Wilson, McNemar, Holm, Cochran–Armitage, GEE, bootstrap, taxonomy) |
| `src/05_analysis.py` | D6: figures fig1–fig7, `reports/results.md`, `reports/tables/*.csv`, gallery HTML |
| `src/06_probe_patch.py` | D7: probe, activation patching, attention knockout → fig8 + `probe_summary.json` (H5) |
| `scripts/run_all_local.py` | Sequential local runner (keeps Windows awake) |
| `scripts/make_kaggle_bundle.py`, `scripts/make_notebooks.py` | Kaggle bundle zip; notebook generators. Edit the generator, **not** the `.ipynb` files |
| `results/<model>.{jsonl,parquet,meta.json}` | Raw per-query outputs. `*_smoke*` and `mock*` are ignored by the analysis |
| `reports/` | `results.md`, `figures/`, `tables/`, `audit/` |

The environment is Windows with an RTX 4050 (6 GB), Python 3.14, torch 2.11 cu128, and transformers 5.23. Use `.venv/Scripts/python` and set `PYTHONIOENCODING=utf-8`. Keys go in a git-ignored `.env` with `HF_TOKEN`, `GEMINI_API_KEY` and `OPENROUTER_API_KEY`. **Never print or commit it.** The original keys were pasted into a chat and should be rotated.

## 3. Current state

| Item | State |
|---|---|
| Real items | 152, from V-COCO (138) and HICO (14). Below the 400 target, logged as a deviation |
| Synthetic items | 800 |
| smolvlm2-2.2b | Full run complete: 29,928 queries |
| llava-ov-0.5b | Full run complete: 29,928 queries |
| qwen2.5-vl-3b (4-bit) | Full run complete: 29,928 queries. The analysis was rerun on real results (see §3b) |
| internvl3-2b | **Not run.** Too slow on 6 GB. Run it on Kaggle following `KAGGLE_RUN.md`, then unzip into `results/` |
| gemini-3.5-flash-lite | Done on 30+30 items, core profile, main block, 840 queries. Uses a text readout, so keep it out of pooled regressions |
| Audit | R1 labels are done. **R2 and R3 are still needed. These are human tasks; an agent cannot do them.** |
| Probe (D7) | Code tested on 6 items, deliberately **deferred** as a TODO |
| Deck (D8) | Not started |

Uncommitted at handover: `results/*` for the three local models, the run log, `HANDOVER.md`, `KAGGLE_RUN.md`, this file and the notebook generator changes. **First action:** check that the run finished (`results/run_all_local.log` ends with `=== analysis end rc=0`), then commit.

## 3b. Results of the completed run (2026-11-15, finished 23:04)

The local run finished cleanly (`run_all_local.log` ends with `analysis end rc=0`). Every result has `n_errors` 0, and each local model answered all 29,928 queries. Run times on the RTX 4050:

| Model | Run time |
|---|---|
| SmolVLM2 | ~1 h |
| LLaVA-OV | 183 min |
| Qwen-3B 4-bit | 301 min |

`reports/results.md`, `reports/figures/` and `reports/tables/` now come from the real models, not the mock model. InternVL is not in yet, and the audit-filtered rerun hasn't been done.

**Reassignment rate (RR):** the share of items in the competence set S_m where the model picks the wrong person (B). From `reports/tables/rr.csv`. RR is 0 for `none` by construction, because S_m is defined on that condition. Gemini ran the core profile only, so it has no `contra_3`.

| Model / dataset | neutral | consistent | contra_1 | contra_2 | contra_3 | contra_4 |
|---|---|---|---|---|---|---|
| SmolVLM2 / real | 0.14 | 0.00 | 0.65 | 1.00 | 1.00 | 1.00 |
| SmolVLM2 / synthetic | 0.06 | 0.00 | 0.52 | 0.79 | 0.75 | 1.00 |
| Qwen-3B / real | 0.06 | 0.00 | 0.45 | 0.79 | 0.62 | 0.83 |
| Qwen-3B / synthetic | 0.00 | 0.00 | 0.07 | 0.27 | 0.10 | 0.39 |
| LLaVA-OV / real | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.96 |
| LLaVA-OV / synthetic | 0.06 | 0.05 | 0.09 | 0.10 | 0.10 | 0.62 |
| Gemini / real (30 items) | 0.00 | 0.00 | 0.00 | 0.00 | – | 0.07 |
| Gemini / synthetic (30 items) | 0.00 | 0.00 | 0.00 | 0.00 | – | 0.00 |

**Competence set size (|S_m|):**

| Model | |S_m| |
|---|---|
| SmolVLM2 | 446 |
| LLaVA-OV | 515 |
| Qwen-3B | 765 |
| Gemini | 59 |

**Capability** (accuracy, from `reports/tables/capability.csv`):

| Model / dataset | Q4 binding with no caption | Q2 action present | Q3 detects a false caption |
|---|---|---|---|
| SmolVLM2 / real | 0.45 | 0.71 | 0.01 |
| SmolVLM2 / synthetic | 0.77 | 0.62 | 0.00 |
| LLaVA-OV / real | 0.55 | 0.83 | 0.04 |
| LLaVA-OV / synthetic | 0.56 | 0.99 | 0.00 |
| Qwen-3B / real | 0.69 | 0.81 | 0.62 |
| Qwen-3B / synthetic | 0.98 | 0.86 | 0.76 |
| Gemini / real | 0.97 | – | 0.97 |
| Gemini / synthetic | 1.00 | – | 1.00 |

The small models almost never flag a false caption as conflicting. LLaVA-OV's Q4 is near chance because of its strong "A" bias; averaging over both option orders removes the bias, but LLaVA-OV then sits near 0.5.

**Hypothesis verdicts** (from `reports/results.md`):

| Hypothesis | Verdict | Details |
|---|---|---|
| **H1** (text causes rebinding) | Supported in 4/8 model×dataset cells | Holm p<.05 against both neutral and none. The non-supporting cells are Gemini (it doesn't flip) and the weak LLaVA cells |
| **H2** (dose-response) | **Strongly supported** | GEE odds ratio per confidence level is 3.46 [2.85, 4.18] on real (p≈5e-37) and 2.22 [2.10, 2.34] on synthetic (p≈3e-171). The trend is positive in 88% of cells and monotone in 62%. `contra_3` often dips below `contra_2`, which could be a quirk of that caption's wording; worth a look |
| **H3** (rebinding, not perception loss) | **Model-dependent** | Dissociation rate (Q2 still correct when Q4 flips): Qwen-3B ≈ 0.51–0.54, so it genuinely rebinds about half the time. SmolVLM2 ≈ 0.00 and LLaVA ≈ 0.02, so their flips come with losing the action. **The report's summary text ("lower bound > .5 in 5/5 cells") contradicts these numbers. Check the H3 verdict logic in `src/analysis.py` / `05_analysis.py` before quoting it** |
| **H4** (awareness doesn't protect) | Holds in 8/8 cells as the report counts them | Qwen-3B is the exception in spirit: when it flags the conflict it rebinds much less (OR 0.03 real, 0.25 synthetic). LLaVA rebinds *more* when it flags (wide CIs). The other CIs are too wide to tell |
| **H5** (probe) | Not run, deferred | |

**Headline for the deck:**

> Small open VLMs follow a contradictory caption about *who* does an action, and the effect grows with how confident the caption sounds. SmolVLM2 hits 100% wrong at the strongest level.
>
> Only Qwen-3B shows true rebinding (it still sees the action) and partial protection from noticing the conflict.
>
> Gemini is essentially immune, though only 30 items were tested.

## 4. Finishing the project (ordered)

1. **Verify the local runs.**
   - For each model, `results/<m>.meta.json` should show `n_errors` 0, and each `.jsonl` should have 29,928 lines.
   - Confirm that `reports/results.md` no longer says "mock".
   - If the analysis step failed, run `python src/05_analysis.py` and read the traceback.
2. **InternVL3-2B on Kaggle.** This is a human step (Kaggle UI) per `KAGGLE_RUN.md`. Afterwards, unzip the results into `results/` and rerun the analysis.
3. **Audit R2/R3.** Done by humans. Then run:
   ```bash
   python src/02_audit_sample.py score
   python src/02_audit_sample.py triage
   python src/05_analysis.py --audited-only
   ```
   Report κ and precision (CVC5), and report results on both all items and audited-only items.
4. **Probe (optional):**
   ```bash
   python src/06_probe_patch.py --model qwen2.5-vl-3b --n 200
   ```
   This takes about 1.5 h. Then rerun `05_analysis.py` so the H5 verdict appears. Ask the user before starting any run longer than about 15 min.
5. **Fill the hypothesis table.** Take it from `reports/results.md` (verdicts per H, per model, per dataset) into the deck.
6. **Build the deck** (section 5 below).

## 5. How to present it (deck, 10–12 slides, about 15 min)

Follow `PLAN.md` §10. Every slide should rest on a figure or number from `reports/`.

| # | Slide | Source |
|---|---|---|
| 1 | Problem: one image with A/B markers plus a `contra_4` caption, and the model's flipped answer | `reports/gallery.html`, fig7 |
| 2 | Gap: "text-over-vision" is known, but *binding* failures have not been separated from *perception* failures | PLAN §1 |
| 3 | Hypotheses H1–H5 and an outcome table (supported / not / inconclusive) | PREREGISTRATION, results.md verdicts |
| 4 | Why the naive HICO-DET version failed: ambiguous multi-person labels | PLAN §0, schema_notes |
| 5 | Dataset construction: CVC1–5 rules and the **funnel** (48,464 → 231 → 196 → 152) | fig1, `funnel.csv` |
| 6 | Data quality: audit κ, precision with Wilson CI | `02_audit_sample.py score` output |
| 7 | Dataset balance: verbs and sources, real vs synthetic | fig2 / tables |
| 8 | Design: conditions × questions, counterbalanced order, logit readout, competence set S_m | diagram |
| 9 | Capability (Q1–Q3 accuracy per model) and letter bias | capability tables |
| 10 | **Main result:** dose-response of the reassignment rate (none → contra_4), forest plot of H1 effects | fig3/fig4 |
| 11 | Failure taxonomy: perception loss vs rebinding vs awareness-but-compliance (H3/H4) | fig5/fig6 |
| 12 | Limitations (152 real items, small models, 4-bit Qwen, Gemini text readout, audit coverage) and future work / probe | PREREGISTRATION deviations |

Talking points:
- **Why this design is clean:** same image across conditions, only the text changes; both answer orders are averaged; models are evaluated on items they get right without a caption (S_m).
- **Real vs synthetic:** synthetic scenes give power, real images give validity. Show they agree.
- **State the deviations openly:** the item count, the detector-based CVC1, and the limited Gemini run. Reviewers respect this.
- **Example smoke numbers** (to be replaced by the full-run figures): Q4 accuracy goes from 0.58–0.83 with no caption down to 0.0–0.3 at `contra_4`.

Format: the user has not chosen one. Ask whether they want PowerPoint (.pptx), Google Slides, or a web deck.

## 6. What more it can do (extensions, none implemented)

- **Scale:** run 7B models (Qwen2.5-VL-7B is already in the config, 4-bit or on Kaggle), more API models once quota allows, and real-item counts closer to 400 (HICO-DET test split, other HOI datasets, relaxing the bystander rule).
- **Mechanistic:**
  - Run the full probe on more models.
  - Do layer-wise patching on image tokens with a text-first prompt order (it is a no-op with the current image-first order).
  - Add an attention-knockout sweep across heads.
  - Do logit-lens on the A/B tokens.
- **Mitigation:** test the existing ablation system prompt ("Answer from the image, not the caption") at scale. Also test chain-of-thought or asking Q3 first, and measure whether conflict detection can gate the answer.
- **Human baseline** on the same items and conditions (PLAN §9).
- **Other bindings:** attributes (colour/size), spatial relations, and more than two people.
- **Release:** package the benchmark as a HF dataset with a leaderboard script.

## 7. Rules for the next agent

- **Long runs need permission.** Ask before any run over about 15 min and give a time estimate. The s/pass figures are in `HANDOVER.md`.
- **Runs resume.** Never delete `results/*.jsonl` to "restart".
- **Keep the prompts fixed.** Changing `configs/verbs.yaml` changes `config_hash` and breaks comparability. Log any change in PREREGISTRATION Deviations.
- **No secrets anywhere visible.** Do not commit `.env`, and do not put keys in notebooks, logs or bundles (`make_kaggle_bundle.py` already excludes them).
- **Commits and pushes** happen only when the user asks.
- **Mock model:** `python src/04_run_models.py --model mock` followed by `python src/05_analysis.py --models mock` is a ~1 min pipeline check that needs no GPU.
- **Known gotchas:**
  - pandas 3 dtype strictness (`audit_keep` is object).
  - Windows cp1252 output errors (set `PYTHONIOENCODING`).
  - Kaggle mounts input at variable depth (the notebook uses `os.walk`).
  - AI Studio Gemini keys return no logprobs.

## 8. Update 2026-11-16 (presentation prep)

- **H3 correction.** §3b misreads H3. `P(Q2 ok | rebind)` is 0.89–1.00 in all 5 measurable cells, so models keep perceiving the action when they rebind: **H3 is supported** (the `results.md` line is right). The "dissociation rate" DR = P(Q2 ok AND Q3 flags conflict | rebind) is low for SmolVLM2/LLaVA because they never flag the conflict (Q3), not because they lose perception.
- **H5 probe run** (Qwen-3B 4-bit, 352 items, `results/probe_run.log`). Probe trained on `none` decodes the true actor at ~0.9 from layer 23 on, but on the 101 output-flipped `contra_3` items it reads the *caption's* person (0.22 [0.15, 0.31] at layer 34). **H5 not supported:** the caption overwrites the actor representation itself. The old verdict rule (max flipped-accuracy over mid layers) picked layers where nothing is decodable; it is now `analysis.h5_verdict` (only layers with clean accuracy ≥ 0.75).
- **`src/07_extras.py`** (no GPU): Beta-Binomial posteriors of RR (fig9, `bayes_rr.csv`, `bayes_effects.csv`), PCA of probe residuals + probe on k PCs (fig10, `pca_probe.csv`), refreshes `probe_summary.json`. PC1 (28%) is exactly real vs synthetic. Writes `reports/results_extras.md`.
- **`src/demo.py`**: local web demo. `python src/demo.py` (live SmolVLM2; custom captions) or `--replay` (recorded answers, no GPU). Port 8800 (launch.json: `demo-live`, `demo-replay` on 8801).
- **PLACEHOLDER audit**: `labels_R2.csv`/`labels_R3.csv` are simulated (`scripts/make_placeholder_audit.py`; `--remove` deletes them). Kappa/precision/`results_audited.md` are not real until humans label. Logged in PREREGISTRATION deviations.
- **Patching/knockout (60 items, all used):** recovery of the true answer when patching clean→corrupt sits on the *entity token* in layers 0–15 (~0.9–1.0), moves to the *question tokens* in layers 18–24 (~0.6), and to the *final token* from layer 27 (0.5 → 0.9 at layer 33). Knockout of final→entity attention changes nothing (Δ≈0): the caption's entity is routed via the question tokens, not read directly. fig8 = `reports/figures/fig8_probe_patch.png`.
- **Order of analysis commands matters:** `05_analysis.py --audited-only` overwrites `reports/tables` and `figures`; always finish with plain `05_analysis.py` then `07_extras.py`.
- Live demo verified 2026-11-16: matches the recorded run to the percent; ~5–6 s per image on the RTX 4050 after a ~35 s model load.
- `02_audit_sample.py score` appends a "CVC5: manual audit kept" row to `data/processed/real/funnel.csv`. With placeholder labels that row is fake: it was reverted (`git checkout -- data/processed/real/funnel.csv`). Rerun `score` only once the real R2/R3 labels are in.
