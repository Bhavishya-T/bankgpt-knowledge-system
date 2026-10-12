#!/usr/bin/env python3
"""Inspect the exact top passages the final submission retrieves."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.run_eval import load_snapshots  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--as-of", help="Optional date in YYYY-MM-DD format")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--submission", default="submission")
    args = parser.parse_args()

    folder = (ROOT / args.submission).resolve()
    sys.path.insert(0, str(folder))
    import importlib.util
    spec = importlib.util.spec_from_file_location("inspected_solution", folder / "solution.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {folder / 'solution.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    system = module.build()
    for snapshot in load_snapshots(ROOT / "snapshots"):
        system.ingest(snapshot)
    trace = system.retrieve(args.question, args.as_of, top_k=args.limit)
    print(f"Question: {args.question}")
    print(f"Snapshot: {trace.get('snapshot_id')} ({trace.get('snapshot_date')})")
    print(f"Candidates: {trace.get('candidate_count', 0)}")
    for passage in trace.get("results", []):
        excerpt = " ".join(passage["text"].split())
        if len(excerpt) > 650:
            excerpt = excerpt[:647] + "..."
        print(f"\n{passage['rank']}. score={passage['score']}")
        print(f"   {passage['title']} — {passage['section']}")
        print(f"   {passage['url']}")
        print(f"   {excerpt}")


if __name__ == "__main__":
    main()
