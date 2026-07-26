# Semantic-Swingers — reproducing the LLMs4OL 2026 data splits

This directory lets anyone **reconstruct the exact train/validation/holdout splits behind our paper**
from the organizers' raw data — deterministically, in any environment, without us redistributing any
benchmark content.

- **`split_ids.json`** — the composition of each split as a list of record **`id`s** (plus an
  `sha256_sorted_ids` fingerprint per split). Metadata only: **no `context` text and no gold triples.**
- **`build_splits.py`** — joins those ids against the organizers' raw release and writes the split
  files, verifying each against its fingerprint.

## 1. Get the organizers' raw data

The splits derive from the official **LLMs4OL 2026** training releases (Task A and Task B):

- Challenge site: https://sites.google.com/view/llms4ol2026
- Flagship (Task A) task page: https://sites.google.com/view/llms4ol2026/flagship-task
- Datasets & evaluation code: https://github.com/sciknoworg/LLMs4OL-Challenge

You need the raw training files (record shape `{"id", "context", "primitive-ontology-triples"}` for
Task A; the Task B training file for the `*_b` splits).

## 2. Rebuild the splits

```bash
python build_splits.py \
    --raw /path/to/train_task_a.json \
    --raw /path/to/train_task_b.json \
    --manifest split_ids.json \
    --out ./data/splits
```

Pass only `--raw .../train_task_a.json` to build just the Task A splits (the `*_b` splits are then
skipped with a message). Each rebuilt split prints `✓ <name> … verified` when its ids match the pinned
fingerprint.

## The splits

| Split | n | Task | Role in the paper |
|---|---:|---|---|
| `train_80` | 3442 | A | Stratified 80% training partition |
| `val_20` | 861 | A | Held-out 20% evaluation set (seen-vocabulary) |
| `train_pool` | 2925 | A | Retrieval / exemplar pool (disjoint from `val_20`, `dev`) |
| `dev` | 517 | A | Tuning set (disjoint from `val_20`) |
| `dev_eval` | 100 | A | Small dev evaluation subset |
| `blind_local` | 297 | A | **Vocabulary-disjoint holdout** — our unseen-ontology generalization proxy |
| `train_pool_b` | 2219 | B | Task B retrieval pool |
| `dev_b` | 555 | B | Task B evaluation set |
| `dev_eval_b` | 100 | B | Task B small dev subset |

`blind_local` is our own vocabulary-disjoint holdout carved from `train_pool` — it is the split we rank
techniques on (unseen − seen gap), not the organizers' blind test set.

## How the splits were originally made (and why we don't ship that script)

The original partition was a stratified 80/20 split by relation type using
`sklearn.model_selection.train_test_split(stratify=…, random_state=1993)`, with the derived splits
(`train_pool`, `dev`, `blind_local`, …) carved from it under leak-free invariants
(e.g. `train_pool ∩ val_20 = ∅`).

We deliberately do **not** ship "re-run the sampler" as the reproduction path: `train_test_split` is
**not deterministic across scikit-learn versions**, so re-running it yields a *different* partition than
the one behind our results. The `id`-manifest above sidesteps that entirely — it reproduces the exact
partition regardless of your library versions.

## What "verified" guarantees

`build_splits.py` reconstructs each split by pulling the raw record for every `id` (a byte-identical
copy of the organizers' record) **in the manifest's order**, then checks `sha256` of the sorted ids
against the pinned fingerprint. So a `✓ verified` split contains exactly the same records, in the same
order, as the split we used. (The on-disk JSON whitespace of your rebuilt files may differ from our
internal copies — that has no effect on loading, scoring, or results.)
