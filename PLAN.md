# Entity–Role Binding Under Contradictory Text: Implementation Plan

Goal: a small, statistically sound study that can be finished fast. **Nothing is trained.** The pipeline is: validated benchmark -> behavioural experiment on open VLMs -> statistics -> (stretch) mechanistic check on one small model.

---

## 0. What went wrong before, and the fix

**Problem:** HICO-DET labels are *incomplete*. They list positive interactions, not exhaustive ones. A caption such as "Person B is holding the umbrella" may be true even though the annotation says Person A holds it. The "contradiction" is then not a contradiction, and every downstream number is contaminated.

**Fix: make the contradiction true by construction.** An item enters the benchmark only if the false caption *cannot* be true. We guarantee that with five *Contradiction Validity Conditions* (CVC), then **measure** the residual label noise with a human audit instead of assuming it is zero.

| # | Condition (applied to the COCO-format annotations of the image) | Why |
|---|---|---|
| CVC1 | Exactly **2** non-crowd `person` boxes in the whole image, each >= 3% of image area, IoU < 0.3 | No third person who could be the "real" holder; the two are separable |
| CVC2 | Exactly **1** instance of the target object class in the COCO annotations **and** in an independent detector run at a deliberately low confidence threshold (e.g. 0.2; YOLOv8/OWLv2/Grounding-DINO). Use the *union* of both sources and require the count to be 1 | Annotations miss small/occluded objects. Counting from two sources and demanding 1 makes a hidden second object unlikely, though not impossible, which is why CVC5 is a full manual check |
| CVC3 | HICO-DET positive (verb, object) pair links the object to exactly one of the two persons | Defines the true actor |
| CVC4 | The *other* person has **no** HICO interaction with that object class, and the verb is on an **exclusive-verb whitelist** | Drops shareable roles (ride bicycle, hold hands, etc.) |
| CVC5 | **Every** final item passes manual review (section 3), judged on: "Could the *other* person plausibly be doing this action too, or is there a second such object?" Anything "yes" or "unclear" is dropped | Automatic filters can't see everything, and a sample can't find the bad items; checking all ~300 takes under an hour split three ways |

Whitelist (start here, trim to whatever yields enough items): `kick sports_ball`, `throw frisbee`, `swing baseball_bat`, `swing tennis_racket`, `brush_with toothbrush`. Choose only actions where the object is physically used by one person at a time. **Dropped because two people can share the object or hold their own: `hold umbrella`, `ride bicycle/motorcycle/horse`, `carry`, `hold cup`, `hold hands`.** Doubles tennis can give each person a racket, which is why the detector count in CVC2 and the manual check in CVC5 matter for racket items; drop racket if it fails audit often. **Check the real schema first** (step D0): the HF dataset's field names decide how CVC1-4 are coded.

**Second dataset to remove label dependence entirely: synthetic scenes** (section 2b), where ground truth is exact by construction. Real = ecological validity, synthetic = clean manipulations. If synthetic drawing eats too much time, ship real-only and list it as a limitation.

**Naming entities without celebrities:** overlay marker labels **A** and **B** on the two people (boxes + letters, set-of-marks style). Which person is "A" is randomised. Secondary reference scheme: "the person on the left/right" (checks that results don't depend on the marker trick).

---

## 1. Research question and pre-registered hypotheses

Write these in the repo *before* running models. This is the single biggest "data science" upgrade over the first deck.

Let the **true actor** be the person who really does the action, and the **distractor** the other person.

- **H1 (text rebinding):** a contradictory caption raises P(model names the distractor) above both the no-text baseline and a neutral-text control.
- **H2 (dose-response):** that rate rises monotonically with linguistic confidence (hedged -> plain -> assertive -> authoritative+repeated).
- **H3 (dissociation, the novel claim):** among rebinding errors, most are *perceptually intact*: the model still recognises the action (Q2 correct) and often flags the conflict (Q3). Compare against the same rate among stable cases.
- **H4 (awareness does not protect):** P(rebind | model flags conflict) is not meaningfully lower than P(rebind | does not flag).
- **H5 (stretch, mechanistic):** the true actor stays linearly decodable from mid-layer activations even when the output flips.

Outcome table, so a null result is still a result:

| Observation | Interpretation |
|---|---|
| Flips, but Q2 also fails | Perception degraded by text. Not a binding failure. |
| Flips, Q2 passes, probe still decodes truth | **Rebinding** (the headline claim) |
| Flips, Q2 passes, probe loses truth | Information overwritten, not just rebound |
| Rarely flips | Models are robust at this scale; report that |

---

## 2. Benchmark design

### 2a. Real subset (primary)
Target: **>= 400 audited items**. If CVC1-4 yield fewer than 400, relax the area threshold to 2% and report the funnel. Expect a steep funnel (47,776 images -> a few hundred to low thousands). **Publish the funnel table**: it is the evidence that you took ground truth seriously.

### 2b. Synthetic subset (secondary)
Procedurally drawn flat-style scenes (PIL or SVG/cairosvg): 2 figures with distinct clothing colours, 4 verbs with fixed pose templates (kick ball, hold umbrella, swing racket, hold flag), randomised left/right, size, background, colours. **Target ~800 scenes.** Ground truth is exact; this is where the dose-response and ablations run cleanly. Budget about half a day for the drawing code.

### 2c. Counterbalancing (every item, every condition)
Randomise and record: which person is A; which side the actor is on; the option order in the question ("A or B" vs "B or A"). Each item is queried **under both option orders** and the logits are averaged, which cancels position/letter bias.

### 2d. Conditions (within-image design: same image, only the text changes)

| Code | Text given with the image |
|---|---|
| `none` | no caption (baseline) |
| `neutral` | content-free caption: "The image shows A and B." (controls for mere presence of text) |
| `consistent` | "A is kicking the ball." (true) |
| `contra_1..4` | "B is kicking the ball." (false), at four confidence levels (below) |

Confidence ladder (the manipulated "strength of contradictory language"):
1. hedged: "I think B might be kicking the ball."
2. plain: "B is kicking the ball."
3. assertive: "B is definitely kicking the ball."
4. authoritative + repeated: "Verified annotation: B is kicking the ball. Confirmed: it is B who kicks the ball."

Ablation (run on a subset): add a system-style instruction "Answer from the image, not the caption" to `contra_2`.

### 2e. Questions (each in a **fresh context**, no chat history, to avoid leakage between probes)
- **Q1 Identity** (visual only, no text): "Which marker is on the person on the left, A or B?" For synthetic: "Who wears the red shirt?" This verifies the model can resolve the *entities*.
- **Q2 Action:** "Is anyone kicking a ball? Yes/No."
- **Q3 Conflict detection:** "Does the caption match the image? Yes/No." (caption-bearing conditions only)
- **Q4 Binding:** "Who is kicking the ball, A or B?"

**Readout is a forced-choice logit comparison**, not generated text: take logits of the tokens `A` vs `B` (or `Yes` vs `No`) at the answer position. No answer parsing, deterministic, and it gives a continuous score (`logit_diff = logit(true) - logit(distractor)`) alongside the 0/1 answer.

### 2f. Query budget (why this is feasible)

Per item: Q4 x 7 conditions x 2 option orders = 14, Q2 x 3, Q3 x 4, Q1 x 1 = about **22 single-token forward passes**.
~1,200 items x 22 = ~26k passes per model. At ~0.15-0.3 s each with low-res images, that's **roughly 1.5-2 h per model**. Parallelise nothing; just run overnight.

---

## 3. Data-quality audit (answers the "ground truth" criticism directly)

1. **Review every candidate item** (about 300, about 100 each). For each, answer: "Is exactly one of A/B doing the action? Who? Could the other person also be doing it, or is there a second such object, even partly hidden? yes/no/unclear." Keep only clean "exactly one, certain" items.
2. **Double-label a shared subset of ~100** by all three people to measure agreement; the rest are single-labelled.
3. Report **Fleiss' kappa** on the shared subset, and the **automatic filter's precision (share of candidates that survived review) with a 95% Wilson CI**. That precision is the funnel's last row.
4. Add **model-disagreement triage** (a confident-learning-style check): items where *every* model gets the no-text Q4 wrong go to manual review, **not** auto-deletion. Auto-deleting hard items would bias the benchmark toward easy ones.
5. Report results on the full set **and** the audited-clean subset. If they agree, the noise didn't matter.

---

## 4. Models (4-5 open VLMs)

| Model | Size | Notes for a 16 GB RTX 5080 |
|---|---|---|
| Qwen2.5-VL-3B-Instruct | 3B | bf16, ~8 GB. Main mechanistic model |
| InternVL3-2B | 2B | fits easily |
| SmolVLM2-2.2B or LLaVA-OneVision-0.5B | small | fast; adds a scale axis |
| Qwen2.5-VL-7B-Instruct | 7B | needs 4-bit/AWQ; stretch |
| LLaVA-1.5/OneVision-7B | 7B | stretch |

The 5080 is Blackwell: use PyTorch >= 2.7 built for CUDA 12.8. If bitsandbytes misbehaves, drop the 7B models rather than burn a day. If there is no GPU at all, free Kaggle/Colab T4 hours cover the small models. **Scale (2B vs 3B vs 7B) is a useful analysis axis** (does susceptibility fall with size?), so keep at least three sizes.

---

## 5. Metrics

Notation: for model *m*, let `S_m` = items where the model is **correct on `none` Q4 (in both option orders) and on Q2**. Rebinding is only meaningful where the model could bind correctly in the first place. This conditioning is what separates "binding failure" from "can't see".

- **Reassignment Rate** `RR(c) = P(model names distractor | i in S_m, condition c)`.
- **Net text effect** `RR(contra) - RR(neutral)`.
- **Entity-Role Binding Stability** `ERBS = 1 - RR(contra)`, reported per confidence level.
- **Dissociation Rate** `DR = P(Q2 correct and Q3 flags conflict | rebinding occurred)`. The headline number for H3.
- **Susceptibility slope:** coefficient of confidence level in the regression below (continuous logit-diff version too).
- **Outcome taxonomy per item** under `contra_2`: {Stable, Rebound-with-intact-perception, Rebound-with-perception-loss, Other}. Show as a Sankey/stacked bar per model.

Also report accuracy for Q1, Q2, Q3 per model as the "capability" panel.

---

## 6. Statistical analysis (the "data science" section of the talk)

1. **Descriptives/EDA:** funnel table; item counts by verb/object; balance of actor side, A/B assignment, person area, inter-person distance; bias in `none` (always answers "A"?).
2. **Primary inference, paired by image:** McNemar's test for `contra` vs `neutral` and vs `none`; effect as odds ratio with **cluster-bootstrap 95% CI** (resample images, 2,000 reps).
3. **Dose-response (H2):** logistic regression / GEE
   `rebind ~ confidence_level + model + verb + scale_of_person_box + actor_side`, with images as clusters (statsmodels `GEE` or `MixedLM`; random intercept per image). Plus a Cochran-Armitage trend test.
4. **H3/H4:** Wilson CI for DR; Fisher / chi-square for rebinding vs conflict-flagged; odds ratio with CI.
5. **Multiple comparisons:** Holm correction across models x conditions. Report *effect sizes + CIs first*, p-values second.
6. **Robustness:** 3 paraphrase templates for the caption and the question, report between-template variance; real vs synthetic; audited-clean vs full; left/right vs A/B referencing.
7. **Power note:** at n = 400 the 95% CI half-width on a proportion is <= ~4.9 pp; at n = 800 <= ~3.5 pp. State this in the deck.
8. **Error analysis:** per-verb heatmap (model x verb), regression of rebinding on visual difficulty features, 20 hand-inspected examples per model.

### Figures list
(1) data funnel; (2) dataset EDA panel; (3) capability panel Q1-Q3 by model; (4) **dose-response curves with CIs**, one line per model; (5) outcome-taxonomy stacked bars; (6) forest plot of odds ratios; (7) model x verb heatmap; (8) (stretch) layer-wise probe accuracy and patching recovery.

---

## 7. Mechanistic analysis (stretch; do only if section 6 is done)

One model only (Qwen2.5-VL-3B). Put the **image first and the text after** in the prompt, so image tokens sit at identical positions across conditions.

1. **Linear probe (cheap, forward passes + sklearn):** cache last-token residual stream at every layer for `none` and `contra_3`; train a logistic probe for "true actor = A/B" on `none` and test on `contra_3`, using **grouped CV by image**. If accuracy stays high at mid layers while the final output flips, that supports H5 (information present, rebound later).
2. **Activation patching:** run the contradictory prompt, patch in activations from the clean (`none`) run at (layer, position) in {image tokens, text-entity token, final token} and measure recovery of `logit_diff`. A sweep over layers x positions gives a heatmap of where the binding is decided.
3. **Attention knockout:** block attention from the final token to the text-entity tokens in a layer window; see whether the flip disappears.

Be explicit in the writeup: these are *correlational / causal-on-one-model* results and may not transfer to other models.

---

## 8. Repo layout and order of work

```
erb2/
  PLAN.md                  # this file + hypotheses frozen before runs
  configs/verbs.yaml       # whitelist, thresholds, templates
  src/00_inspect_schema.py # D0: look at HICO-DET-coco fields; decide how CVC1-4 are coded
  src/01_build_real.py     # CVC filters -> items.parquet + funnel.csv
  src/02_audit_sample.py   # stratified 200 sample -> audit.csv; kappa + Wilson
  src/03_make_synthetic.py # scenes + exact ground truth
  src/04_run_models.py     # per model: all conditions x questions x orders -> results.parquet
  src/05_analysis.ipynb    # stats + all figures
  src/06_probe_patch.py    # stretch
```

Suggested order (each step has a done-criterion):

| Step | Output | Done when |
|---|---|---|
| D0 | schema notes | you know where person boxes, object boxes, verbs live |
| D1 | `01_build_real.py` + funnel | >= 400 candidates, distribution plots |
| D2 | audit | kappa + precision CI computed |
| D3 | `04_run_models.py` on 50 items, smallest model | end-to-end, per-pass time measured |
| D4 | synthetic generator | 800 scenes, spot-checked |
| D5 | full runs, all models (overnight) | results.parquet per model |
| D6 | analysis notebook | all 7 figures, hypothesis table filled |
| D7 | stretch: probe/patching | heatmap on 1 model |
| D8 | deck | see below |

Always **test the pipeline on 50 items first** and time it before committing to a full run. Log model name, commit, prompt templates, and `torch`/`transformers` versions in the results file.

---

## 9. Scope: what to claim and what to defer

- **Do now:** benchmark + audit, behavioural experiment on 3-5 models, full statistics, probe if time allows.
- **Future work:** full activation patching across models, 7B+ models, larger scale, human baseline for the *text-over-vision* effect, fine-tuning to reduce rebinding.
- **Limitations to state up front:** marker overlay is an artificial identifier; synthetic scenes are not natural; mechanistic claims come from a single model; forced-choice logit readout is not free-form generation.

---

## 10. Deck outline (10-12 slides, built around evidence)

1. Problem + one example image with the contradiction
2. Prior work gap (text-over-vision exists; binding-vs-perception not separated)
3. Hypotheses H1-H5 and the outcome table
4. **Why the first attempt failed:** the HICO-DET ambiguity example
5. **Dataset construction:** CVC rules + the funnel figure
6. **Data quality:** audit, kappa, precision CI
7. EDA/balance of the final dataset
8. Experimental design (conditions x questions, counterbalancing)
9. Capability results (Q1-Q3)
10. **Main result:** dose-response curves + forest plot
11. Taxonomy of failures (perception loss vs rebinding) + H3/H4 numbers
12. Mechanistic preview or future work, limitations

---

## 11. One-day scope and fallbacks if there is no 5080

**One-day cut:** real subset only (~300 items, ~100-item audit), 3 small models, 3 confidence levels, no synthetic scenes, no mechanistic part (future work).

**Write the runner behind one interface** so the backend is swappable:
`ask(image, prompt, options) -> {answer, p_options or None}`. Backends: `hf_local`, `gemini_api`, `openrouter`.

**Fallback order**
1. **Free cloud GPU (Kaggle notebook T4s, ~30 GPU-h/week, or Colab free T4).** Same open models and same code as the 5080 plan. Use fp16 (T4 has no bf16) and `attn_implementation="sdpa"`. Best option: it keeps the logit readout.
2. **Gemini API free tier (AI Studio): `gemini-2.5-flash` / `gemini-2.5-flash-lite`, plus `gemma-3-27b-it` (also vision).** Behavioural experiment only. Use temperature 0, "Answer with only A or B", parse the letter. Request `responseLogprobs` if the model supports it; otherwise score 0/1 only and drop logit-diff plots for this model.
3. **OpenRouter free vision models.** Availability and rate limits change; treat as a bonus.
4. **CPU only:** SmolVLM-256M/500M, small item count. Slow but works.

**API budget.** Free-tier limits are tight (requests/min and especially requests/day) and change; **check the current quotas in AI Studio before planning**. Trim the API run to the core calls per item: Q4 under `none`, `neutral`, `consistent`, `contra_1`, `contra_2`, `contra_4`, each in both option orders, plus Q2 and Q3 under `contra_2` = about 14 calls. At 150 items that's ~2.1k calls, which may need to be spread over several days on a strict daily cap. If the cap blocks you, pay a few dollars rather than using several accounts or keys.

**Reporting.** Label the API model "closed-model reference point". Do not mix its numbers into the open-model regression without a model fixed effect, and note that its readout is text, not logits.
