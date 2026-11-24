# ERB2 · Entity–Role Binding Under Contradictory Text

When a caption names the wrong person, does a vision-language model change who it thinks performed the action? ERB2 tests this by holding a two-person image fixed, varying the caption, and asking which marked person performed the action. The benchmark contains 152 real images (138 V-COCO and 14 HICO-DET) and 800 synthetic scenes.

## Replay the experiment

From the repository root, install the replay dependencies and start the guided demo.

```sh
python -m pip install numpy pandas pyarrow pyyaml pillow
python src/demo_app.py --replay
```

Open [localhost:8810](http://127.0.0.1:8810). Replay uses saved answers and report tables. It needs neither a GPU nor API credentials. The walkthrough notebook is [ERB2_walkthrough.ipynb](notebooks/ERB2_walkthrough.ipynb).

## Code map

| Stage | Files |
|---|---|
| Dataset | `inspect_schema.py` checks the source annotations. `build_real_dataset.py` filters the real images, `generate_synthetic.py` draws controlled scenes, and `audit_dataset.py` prepares and scores human reviews. |
| Model evaluation | `common.py` builds queries from `configs/verbs.yaml`. `backends.py` handles model input and reads out answers. `run_models.py` saves one result per query. |
| Analysis | `analysis.py` defines the metrics and tests. `analyze_results.py` builds the reports using `reporting.py`. `probe_model.py` runs the linear probe and activation patching. `bayes_and_pca.py` runs the exploratory analyses. |
| Demo | `demo_app.py` serves the data and recorded answers through `demo_app.html`. |

The Python files live in `src/`. Prepared items are in `data/processed/`, model outputs in `results/`, and figures and tables in `reports/`.

## Reproduce the benchmark

Install `requirements.txt` for the full pipeline. Local inference and probing require a compatible PyTorch installation. API backends read credentials from the environment or a local `.env` file. See `.env.example` for the variable names.

| Task | Command |
|---|---|
| Inspect schemas | `python src/inspect_schema.py` |
| Build real or synthetic items | `python src/build_real_dataset.py` or `python src/generate_synthetic.py` |
| Prepare and score human reviews | `python src/audit_dataset.py prepare` and, after collecting labels, `python src/audit_dataset.py score` |
| Run a model | `python src/run_models.py --model <name>` |
| Analyze saved results | `python src/analyze_results.py` |
| Probe Qwen | `python src/probe_model.py --model qwen2.5-vl-3b` |
| Run the Bayesian analysis and PCA | `python src/bayes_and_pca.py` (PCA uses the residuals saved by the probe) |

See [PREREGISTRATION.md](PREREGISTRATION.md) for the hypotheses, decision rules, and deviations. The saved R2 and R3 audit labels are simulated. The audit summary and audited-only report demonstrate the workflow. Use the [main results report](reports/results.md) for the benchmark findings.
