# BCA RAG Watermark Experiments

This folder contains the release version of the main aligned-control experiments:

- Ward baseline: single-document query generation.
- BCA: paired-document anchor planning plus comparison query generation.
- ReAct-style RAG: retrieve, compress evidence, generate answer.
- KGW detection and pooled result analysis.

The code expects the project `MarkLLM/` directory to be available either next to this
folder or inside it.

## Setup

Create a `.env` file with the keys you use:

```bash
DEEPSEEK_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

Install with uv:

```bash
cd release
uv sync
```

## Run Farad Main Experiment

From the repository root:

```bash
uv run python release/run_experiment.py \
  --method both \
  --token-budget 16384 \
  --work-dir Data/scale_4096_aligned_query_sets_rerun_a1 \
  --wm-dir Data/watermarked_texts \
  --clean-dir Data/original_texts \
  --out-dir Data0518 \
  --run-id farad_release \
  --backbone-model deepseek-v4-flash \
  --backbone-temperature 0 \
  --query-style anchor_comparison \
  --pair-strategy hybrid_centered_orthogonal \
  --pair-alpha 1 \
  --pair-count 130 \
  --seed 42 \
  --rotate-inputs
```

Analyze pooled z-scores:

```bash
uv run python release/analyze_results.py \
  Data0518/scaled_ward_aligned_16384_farad_release.json \
  Data0518/scaled_contrastive_hybrid_centered_orthogonal_a1_anchor_comparison_deepseek-v4-flash_aligned_16384_farad_release.json \
  --budget 16384 \
  --device cuda:1 \
  --out Data0518/farad_release_pooled_summary.json
```

