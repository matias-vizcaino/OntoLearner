#!/usr/bin/env python3
"""Reconstruct the Semantic-Swingers LLMs4OL 2026 data splits from the organizers' raw data.

Why this exists
---------------
Our splits were originally produced by a stratified ``sklearn.train_test_split`` (seed 1993). That
sampler is **not deterministic across sklearn versions**, so re-running it would yield a *different*
partition than the one behind our paper. To make the exact splits reproducible in any environment,
we instead publish the split *composition* as record ``id`` lists (``split_ids.json``) and rebuild the
files by joining those ids against the organizers' raw release. This is deterministic and ships no
benchmark content.

Usage
-----
Download the organizers' raw data first (see splits/README.md for links), then:

    python build_splits.py \
        --raw data/raw/task_a/train_task_a.json \
        --raw data/raw/task_b/train_task_b.json \
        --manifest split_ids.json \
        --out data/splits

Each requested split is written to ``<out>/<name>.json`` and its ``sha256_sorted_ids`` is checked
against the pinned fingerprint. A split whose ids are not fully covered by the provided raw files is
skipped with a message (e.g. omit ``--raw ...task_b...`` and only the Task A splits are built).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _load_records(path: Path) -> dict[str, dict]:
    """Index a raw release file by record id. Accepts a list of records or an id->record dict."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else list(data.values())
    index: dict[str, dict] = {}
    for r in records:
        rid = r.get("id")
        if rid is None:
            raise SystemExit(f"{path}: a record has no 'id' field — wrong file?")
        index[rid] = r
    return index


def _fingerprint(ids: list[str]) -> str:
    """sha256 of the newline-joined *sorted* ids (matches the team repo's verifier)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", action="append", required=True, type=Path,
                    help="organizers' raw release file (repeatable: pass Task A and/or Task B)")
    ap.add_argument("--manifest", type=Path, default=Path(__file__).with_name("split_ids.json"),
                    help="split id manifest (default: split_ids.json next to this script)")
    ap.add_argument("--out", type=Path, default=Path("data/splits"),
                    help="output directory for reconstructed split files")
    ap.add_argument("--only", help="reconstruct just this split name")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    splits = manifest["splits"]

    index: dict[str, dict] = {}
    for raw_path in args.raw:
        if not raw_path.exists():
            raise SystemExit(f"raw file not found: {raw_path}")
        part = _load_records(raw_path)
        overlap = index.keys() & part.keys()
        if overlap:
            print(f"! {len(overlap)} ids appear in more than one raw file; keeping the later one",
                  file=sys.stderr)
        index.update(part)
    print(f"loaded {len(index)} raw records from {len(args.raw)} file(s)\n")

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = {args.only: splits[args.only]} if args.only else splits
    built, skipped, failed = 0, 0, 0
    for name, spec in wanted.items():
        ids = spec["ids"]
        missing = [i for i in ids if i not in index]
        if missing:
            print(f"- {name:14s} SKIP ({len(missing)}/{len(ids)} ids not in provided raw files)")
            skipped += 1
            continue
        records = [index[i] for i in ids]                       # manifest order == original order
        got = _fingerprint(ids)
        ok = got == spec["sha256_sorted_ids"]
        (args.out / f"{name}.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{'✓' if ok else '✗'} {name:14s} n={len(records):5d} "
              f"{'verified' if ok else 'FINGERPRINT MISMATCH — do not trust'}")
        built += 1
        failed += not ok

    print(f"\nbuilt {built}, skipped {skipped}, fingerprint-failures {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
