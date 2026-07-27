# Semantic-Swingers — LLMs4OL 2026 companion notebook

This branch is **supplementary reproduction material** for the Semantic-Swingers submission to
[LLMs4OL 2026](https://sites.google.com/view/llms4ol2026). It is **not** part of the library
integration and is **not** proposed for merge upstream — that is the pull request below.

- **Methodological diagram:** [`METHODOLOGY.md`](../METHODOLOGY.md) — the shared two-stage pipeline and per-task specialization.
- **Native integration PR:** [sciknoworg/OntoLearner#338](https://github.com/sciknoworg/OntoLearner/pull/338)
- **Submission announcement issue:** [sciknoworg/OntoLearner#339](https://github.com/sciknoworg/OntoLearner/issues/339)
- **Fine-tuned weights (Hugging Face Hub, public):**
  - [`datagero/qwen3.5-9b-ontology-extraction-raft`](https://huggingface.co/datagero/qwen3.5-9b-ontology-extraction-raft) — Task A, retrieval-aware FT (k=10), CUDA/peft
  - [`datagero/qwen3.5-9b-ontology-extraction-baseft`](https://huggingface.co/datagero/qwen3.5-9b-ontology-extraction-baseft) — Task A, base FT (k=0), CUDA/peft
  - [`datagero/qwen3.5-9b-ontology-extraction-baseft-mlx`](https://huggingface.co/datagero/qwen3.5-9b-ontology-extraction-baseft-mlx) — Task A, base FT, Apple-Silicon/MLX
  - [`datagero/taxonomy-structural-matrix-1024-mxbai`](https://huggingface.co/datagero/taxonomy-structural-matrix-1024-mxbai) — Task C, bilinear structural matrix `W` (1024-D, mxbai)

## What `pipeline_ontolearner.ipynb` shows

An end-to-end walk through all three tasks using the Semantic-Swingers learners added in PR #338:

- **Task A (flagship, text2onto):** retrieval-augmented generation with a LoRA fine-tuned Qwen3.5-9B.
- **Task B (reuse, term typing):** closed-vocabulary typing with a swappable selector.
- **Task C (taxonomy discovery):** retrieval-first taxonomy induction with a swappable parent selector.

The learners run on paid (`openai`) or fully-open/offline (`ollama`, local `peft`/`mlx`) backends —
selectable via `LLM_PROVIDER` — so the pipeline is reproducible without a paid API (an open model is
the fallback for Tasks B/C; Task A uses the published adapters).

The committed cell outputs are from a real run and are kept for reference.

## Requirements not included in this branch (by design)

This notebook was authored in the Semantic-Swingers **team** repository and expects three things that
are deliberately **not** redistributed here:

1. **Benchmark data — not included.** The notebook reads `data/splits/*.json`
   (`blind_local`, `val_20`, `train_pool`, `dev_b`) under `DATA_HOME`. These are derived from the
   organizers' LLMs4OL datasets and are **not** committed (we never redistribute benchmark data).
   Point `DATA_HOME` / `LLMS4OL_REPO_ROOT` at a checkout that has them, in the expected shape.
2. **Evaluation metrics.** Scoring cells import `graph_similarity_v2` (the organizers' graph-similarity
   metric), `score_task_b`, and `task_c_gold_set`. The official evaluation scripts live in the
   [LLMs4OL metrics directory](https://sites.google.com/view/llms4ol2026); make them importable
   (e.g. on `PYTHONPATH`) before running the scoring cells.
3. **The learners themselves.** `pip install` the fork (this repository / branch of PR #338) so
   `from ontolearner ...` resolves.

## Running it

Set the environment knobs the notebook reads, then execute top-to-bottom with small `N_*` first:

| Variable | Purpose |
|---|---|
| `LLMS4OL_REPO_ROOT`, `DATA_HOME` | where the split files + eval modules live |
| `LLM_PROVIDER` | `local` (ollama/open) or `paid` (openai) |
| `N_TASK_A`, `N_TASK_B`, `N_TYPES_C`, … | per-task document counts (keep small to smoke-test) |
| `RUN_FT`, `FT_ADAPTER`, `FT_SPLIT`, `FT_DEVICE` | enable the Task A fine-tuned path and pick an adapter |

> **Task A needs a CUDA GPU for a fast run.** On Apple Silicon the transformers/PEFT path falls back to
> CPU (minutes per document); use the MLX adapter (`baseft-mlx`) on a Mac, or a CUDA host for the
> peft adapters. Tasks B and C run comfortably on CPU/laptop with an open model.
