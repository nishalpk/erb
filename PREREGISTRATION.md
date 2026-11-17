# Pre-registration: Entity–Role Binding Under Contradictory Text

Frozen on 2026-11-13, before any model was queried. Analysis code reads these definitions; changes after
data collection must be listed under "Deviations" with a date and reason.

## Definitions

- **Item**: one image with two marked people (A, B). Exactly one of them, the **true actor**, does the
  action with the single target object. The other is the **distractor**.
- **Readout**: forced choice. For local models: logits of the option tokens at the answer position,
  renormalised over the options. For API models: the parsed letter (0/1), and token logprobs if the API returns them.
- **Order averaging**: every Q4 is asked with options in both orders ("A or B", "B or A").
  `p_true(item, cond) = mean over orders of p(true actor)`. The model **names the distractor** when `p_true < 0.5`.
- **Competence set** `S_m`: items where model *m* picks the true actor under `none` in **both** orders
  *and* answers Q2 (action present) correctly under `none`. For the API core profile, which has no `none` Q2,
  only the Q4 part is used, and the report says so.

## Hypotheses

| ID | Statement | Test | Decision rule |
|---|---|---|---|
| H1 | Contradictory caption raises P(name distractor) above `none` and above `neutral` | McNemar, paired by item, on `S_m`, for `contra_2` vs `neutral` and vs `none`; OR with 2,000-rep cluster-bootstrap CI | Supported for model *m* if Holm-adjusted p < .05 for both comparisons and the RR difference is > 0 |
| H2 | RR rises monotonically with confidence level 1→4 | Cochran–Armitage trend per model; pooled GEE logistic (exchangeable, clusters = item) with the level as a linear term | Supported if the GEE slope > 0 with p < .05 and per-model trends are positive in most models |
| H3 | Most rebinding errors are perceptually intact (Q2 correct) and many flag the conflict (Q3 = "No") | Dissociation rate `DR = P(Q2 ok ∧ Q3 flags │ rebind)` under `contra_2`, Wilson CI; `P(Q2 ok │ rebind)` compared with `P(Q2 ok │ stable)` | "Most" = lower Wilson bound of `P(Q2 ok │ rebind)` > 0.5 |
| H4 | Flagging the conflict does not protect against rebinding | Fisher exact on 2×2 (flag × rebind) under `contra_2`; OR + CI | Supported if the OR's 95% CI includes 1 **or** the OR is > 0.5 (no strong protection). Report the OR in either case |
| H5 | (stretch) True actor is linearly decodable from mid-layer activations under `contra_3` even when the output flips | Logistic probe trained on `none`, tested on `contra_3`, CV grouped by image, one model | Supported if accuracy on *flipped* items at the best mid layer has a Wilson lower bound > 0.5 |

Multiple comparisons: Holm across models × comparisons in H1. Effect sizes and CIs are reported before p-values.

## Outcome taxonomy (per item under `contra_2`, on `S_m`)

| Label | Rule |
|---|---|
| Stable | names true actor |
| Rebound, perception intact | names distractor ∧ Q2 correct under `contra_2` |
| Rebound, perception lost | names distractor ∧ Q2 wrong under `contra_2` |
| Other | order-inconsistent (`p_true` exactly 0.5 for text-readout models) |

## Planned robustness checks
3 caption paraphrases × 3 question paraphrases (subset), real vs synthetic, audited-clean vs full,
A/B markers vs left/right referencing, the "answer from the image" instruction ablation.

## Deviations
- 2026-11-13: The HF parquet `zhimeng/hico_det` stores only one human–object pair per HOI category, so it cannot support CVC1/3/4. Annotations are taken from the full HAKE instance-level release of HICO-DET **train** (38,118 images). The test split is not used.
- 2026-11-13: HICO-DET boxes only people who take part in an annotated interaction, so CVC1 also uses the detector: no unannotated person (conf ≥ 0.40, area ≥ 1%) may be present.
- 2026-11-13: Activation patching uses `consistent` → `contra_2` as the clean/corrupt pair (identical token length, differing only in the entity letter) rather than `none` → contra, whose prompts have different lengths.
- 2026-11-13: Added a Q2 *foil* (a whitelisted action absent from the image) under `none`, which measures yes-bias in Q2.
- 2026-11-13: Closed-model reference is `gemini-3.5-flash-lite` (text readout, no logprobs available on AI Studio keys); `gemini-2.5-flash-lite` is closed to new users and `gemini-2.5-flash` has a ~20 requests/day free quota.
- 2026-11-13: The real subset has 152 automatic candidates (V-COCO 138, HICO 14), below the 400 target; the synthetic subset (800) carries power for H1/H2, and real-subset results are reported with their wider CIs.
- 2026-11-13: Visual-token caps per model (`image_processor` / `max_image_side` in the config) so runs fit a 6 GB GPU; recorded in each results meta file.
- 2026-11-14: The Gemini reference run uses `--limit 30` (30 real + 30 synthetic items, 840 queries) instead of 75 per dataset, to stay within the free-tier daily quota. It can be extended later with `--limit 75`; completed queries are kept.
- 2026-11-16: **Placeholder audit labels.** `reports/audit/labels_R2.csv` and `labels_R3.csv` are SIMULATED by `scripts/make_placeholder_audit.py` (R1 copied with 8% resampling on shared items) so the audited-only pipeline can run. Kappa, precision and `results_audited.md` are not real until the human R2/R3 labels replace them; `02_audit_sample.py score` prints a warning banner while placeholders are present.
- 2026-11-16: Added exploratory (not pre-registered) analyses in `src/07_extras.py`: Beta(1,1)-Binomial posteriors of the reassignment rate with Monte-Carlo posterior probabilities, and PCA of the probe's hidden states with a logistic probe on the first k components.
- 2026-11-16: H5 probe/patching run on Qwen2.5-VL-3B (4-bit), 200 items per dataset requested (152 real available), 60 items for patching/knockout. The probe now also caches residuals to `results/probe_resid_<model>.npz` (git-ignored).
