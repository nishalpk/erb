# When Captions Lie: speaker script (15 min, 3 speakers)

| Speaker | Slides | Rubric part | Time |
|---|---|---|---|
| **Speaker 1** | 1–7 | Introduction, Objectives, Block diagram | ~5 min |
| **Speaker 2** | 8–10 | Dataset, Configurations, **Live demo** | ~5.5 min (demo ~4) |
| **Speaker 3** | 11–17 | Results, Analysis, Conclusion | ~5 min |

Handovers are marked with ➜. Slide 18 (References) is shown at the end but not read out.

---

## SPEAKER 1: Introduction → Block diagram

### Slide 1: Title (15 s)
- Good morning. Our project is **"When Captions Lie"**.
- The question: do vision-language models trust the **text** they're given over **what they actually see**?
- Introduce the team in one line.

### Slide 2: Introduction (50 s)
- This is a real photo. **Person A** is throwing the frisbee.
- With no caption, Qwen2.5-VL picks A with **87%** probability.
- We add one false sentence: "B is throwing the frisbee." It drops to **37%**, so the model now names B.
- The image never changed, only the text did.
- Why it matters: these models auto-caption datasets, write alt-text for blind users and do moderation. The text they receive is often wrong.

### Slide 3: Novelty (40 s)
- Past work reports a single accuracy drop when text misleads a model.
- That can't tell apart two failures:
  - the model **stops seeing** the action, or
  - it sees it but **gives it to the wrong person**.
- We isolate the second, which we call a **binding failure**, and treat it as a data-science problem:
  - a curated benchmark
  - captions with graded confidence
  - control questions
  - pre-registered hypotheses
  - a look inside the model

### Slide 4: Objectives (40 s)
- Five objectives. Each maps to a hypothesis and a course method.
- **O1:** build a clean benchmark (data wrangling, inter-rater agreement).
- **O2 / H1:** does a false caption flip *who*? (McNemar test)
- **O3 / H2:** does it grow with confidence? (trend test, logistic regression, Bayesian)
- **O4 / H3–H4:** wrong person or lost perception? Does noticing help?
- **O5 / H5:** does the truth survive inside the model? (probe, PCA)

### Slide 5: Block diagram (60 s)
- Three stages, read row by row.
- **Data:**
  - 48,464 raw labelled photos
  - validity rules plus an object detector
  - result: 152 real + 800 synthetic images, with the two people marked A and B
- **Experiment:**
  - a query generator crosses 7 captions × 4 questions × 2 answer orders
  - 3 open models on a laptop GPU, ~30,000 queries each
  - we read the probability straight from the model's output scores (logits)
- **Analysis:**
  - statistics (CIs, McNemar, GEE, Bayesian)
  - inside-the-model analysis (probe, PCA, patching)
  - outputs: verdicts, figures, demo

### Slide 6: Experimental design (45 s)
- **Left:** the caption ladder.
  - Baselines: none, neutral and true.
  - Then four lies of rising confidence, from "I think B might be…" to "Verified annotation: B is…".
- **Right:** four questions, each asked in a fresh context.
  - **Q4** is the key question: "Who is throwing, A or B?"
  - **Q2** checks the action is still seen.
  - **Q3** checks whether the model notices the caption is wrong.

### Slide 7: How we measure (50 s)
- We never parse free text. We read probabilities.
- **(1)** Softmax over the logits for "A" and "B".
- **(2)** Ask in both orders and average. This cancels letter bias: LLaVA says "A" 98% of the time.
- **(3)** Competence set: only score images the model gets right without any caption.
- **(4)** Reassignment rate: the share of those images where the true person's probability falls below 0.5.
- This is the number on every results slide.
- ➜ *"Now [Speaker 2] will show the data and the system live."*

---

## SPEAKER 2: Dataset, Configurations, Demo

### Slide 8: Dataset (50 s)
- Most action photos are ambiguous: crowds, several objects, both people acting.
- We filter step by step (log-scale chart). Only **0.3%** survive: **152 real photos**.
- Each rule guarantees a clean question: exactly two separable people, one object, and the action done by one person only.
- We add **800 synthetic scenes** for statistical power: perfectly balanced, ±3.5 pp confidence intervals versus ±7.9 pp on real.
- A/B labels and left/right positions are balanced, so always answering "A" can't win.

### Slide 9: Configurations (35 s)
- Three open models: SmolVLM2-2.2B, LLaVA-OneVision-0.5B and Qwen2.5-VL-3B (4-bit).
- All on a 6 GB laptop GPU: **29,928 queries each, zero errors**.
- Prompts frozen and pre-registered. Statistics validated on a mock model with known effects.

### Slide 10: Live demo (~4 min): `src/demo_app.py`, 7 steps = the block diagram
- *(Before the talk: start `.venv/Scripts/python src/demo_app.py` and open http://127.0.0.1:8810. Qwen takes about 30 s to load. Use the ← → keys to move between steps.)*
- *(Image selector: keep the **⚠ fooled · real: throwing the frisbee** item. The selector lists images this model gets fooled on first.)*
- **Step 1, Data:** "This is the raw COCO photo with its original V-COCO labels. The orange person is labelled *throw*. These rules all pass: exactly two people, one frisbee, only one thrower. Then we draw the A/B boxes."
- **Step 2, Prompts:** "Seven captions, from none to 'Verified annotation', and four questions. That's 24 queries for this one image."
- **Step 3, Run:** click **▶ Run all 24 queries**. "Each row is one forward pass on the GPU. Watch the 'who' rows turn red as the lie gets stronger, while 'Is anyone throwing?' stays Yes."
- **Step 4, Readout:** "Here is the model's actual next-token distribution: 'B' 70%, 'A' 30%. We keep the two letters, softmax, then average both answer orders, giving p_true = 37%. That's below 50%, so it names the wrong person."
- **Step 5, Score:**
  - "The image counts because the model gets it right with no caption, in both orders."
  - The bars go from 87% to 37%.
  - Q2 = Yes: it still sees the throw. Q3 = Yes: it didn't notice the lie.
  - **Ask the audience for a caption**, type it and run it.
- **Step 7, Inside** (skip step 6; Speaker 3 covers it with the slides): click **▶ Probe this image**. "Without a caption the probe reads A. With the lie it reads B, so inside the model the lie has overwritten the truth."
- *Backup if the GPU fails:* `.venv/Scripts/python src/demo_app.py --replay` gives the same 7 steps from the recorded run. Step 4's full-vocabulary chart, custom captions and the live probe need the GPU.
- ➜ *"[Speaker 3] will now show what the full experiment found."*

---

## SPEAKER 3: Results, Analysis, Conclusion

### Slide 11: Dose-response, H2 (50 s)
- x-axis: caption condition. y-axis: reassignment rate.
- Baselines near 0%. Lies push the rate up, and "verified" is strongest.
- **SmolVLM2 hits 100%.**
- GEE logistic regression, clustered by image: each confidence step multiplies the odds of error by **3.46** on real images and **2.22** on synthetic.
- **H2 supported.**

### Slide 12: Paired test, H1 (40 s)
- The same image with and without the caption gives paired data, so we use **McNemar**: only the pairs that changed count.
- Holm correction for multiple tests; bootstrap CIs.
- **SmolVLM2 and Qwen flip significantly** on both datasets (+27 to +100 pp).
- LLaVA resists the plain lie.
- **H1 supported** for 2 of 3 models.

### Slide 13: Bayesian analysis (35 s)
- Each answer is a Bernoulli trial. With a Beta(1,1) prior, the posterior is Beta(1+k, 1+n−k).
- The curves slide right as the lie gets more confident.
- P(θ_verified > θ_neutral) = **1.00** for all three models.

### Slide 14: H3 & H4 (40 s)
- **Red** = wrong person but the action still seen. This dominates.
- Orange (action lost) is tiny.
- P(Q2 = yes | rebind) = **0.89–1.00**, so it's **rebinding, not blindness. H3 supported.**
- **H4:** the small models flag the lie in only 0–0.4% of cases. Qwen notices more often, and noticing partly protects it.

### Slide 15: Inside the model, H5 (45 s)
- We trained a logistic-regression probe on Qwen's hidden states at every layer.
- With no caption it reads the true actor at **90%** from layer 23 on.
- On flipped items it reads the caption's person: only **22%** true actor.
- **H5 not supported:** the truth is **overwritten**, not hidden.
- Patching shows the route: caption token → question tokens → final token.
- PCA: 57 of 2,048 dimensions hold 90% of the variance.

### Slide 16: Analysis & key observations (40 s)
- Scorecard: H1, H2 and H3 supported; H4 partly; H5 rejected.
- **Real photos are harder:** Qwen 79% vs 27% on synthetic.
- **Size isn't safety:** LLaVA (smallest) resists until "verified", then follows 96%.
- **Prompting isn't a fix:** "answer from the image" helps one model and hurts another.

### Slide 17: Conclusion (30 s)
- One confident sentence can override what a small VLM sees.
- It still sees the action but gives it to the wrong person, and internally the truth is overwritten.
- Limits: 152 real images, models ≤ 3B, audit in progress.
- Next steps: 7B models and a human baseline.
- **"Thank you. We're happy to take questions."**

---

## Q&A cheat-sheet (anyone)
- **Why no big/API models?**
  - We ran Gemini on only 60 items (free-tier limit), with text-only output.
  - That isn't comparable with the others, so we left it out.
- **Why a probability readout, not text?** It's deterministic, avoids answer-parsing errors, and order averaging removes letter bias.
- **Why only 152 real images?** Strict validity rules. Every one has an unambiguous single actor; synthetic data supplies the power.
- **Why McNemar and not a chi-square test of independence?** The same image is measured twice, so the data are paired.
- **Why GEE?** Repeated measures per image are correlated. GEE clusters by image.
