#!/usr/bin/env python3
"""Measure whether top retrieved passages contain labeled facts and sources."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from run_eval import load_snapshots  # noqa: E402
from scoring import contains  # noqa: E402


def load_system(folder: Path):
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("retrieval_solution", folder / "solution.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {folder / 'solution.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=Path("submission"))
    parser.add_argument("--eval", type=Path, default=Path("eval/public_eval.jsonl"))
    parser.add_argument("--snapshots", type=Path, default=Path("snapshots"))
    parser.add_argument("--out", type=Path, default=Path("results/retrieval-audit"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    items = [json.loads(line) for line in args.eval.read_text().splitlines() if line.strip()]
    system = load_system(args.submission)
    for snapshot in load_snapshots(args.snapshots):
        print(f"ingest {snapshot.snapshot_id} ({len(snapshot.pages)} pages)", flush=True)
        system.ingest(snapshot)

    records, totals = [], defaultdict(lambda: {"n": 0, "source_n": 0, "source_hit": 0, "fact_n": 0, "fact_hit": 0})
    for item in items:
        trace = system.retrieve(item["question"], item.get("as_of"), top_k=args.top_k)
        passages = trace.get("results", [])
        evidence = "\n".join(p["text"] for p in passages)
        source_hit = any(
            expected.rstrip("/").lower().split("#")[0] in passage["url"].lower()
            for expected in item.get("expected_sources", []) for passage in passages
        )
        facts = item.get("must_include", [])
        fact_hit = bool(facts) and all(contains(evidence, fact) for fact in facts)
        row = totals[item["category"]]
        row["n"] += 1
        if item.get("expected_sources"):
            row["source_n"] += 1
            row["source_hit"] += int(source_hit)
        if facts:
            row["fact_n"] += 1
            row["fact_hit"] += int(fact_hit)
        records.append({
            "id": item["id"], "category": item["category"], "question": item["question"],
            "top_k": args.top_k, "source_hit": source_hit, "facts_present": fact_hit,
            "trace": trace,
        })
    report = {
        category: {
            "questions": row["n"],
            "source_labeled": row["source_n"],
            "source_recall": round(row["source_hit"] / row["source_n"], 4) if row["source_n"] else None,
            "fact_labeled": row["fact_n"],
            "fact_recall": round(row["fact_hit"] / row["fact_n"], 4) if row["fact_n"] else None,
        }
        for category, row in sorted(totals.items())
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "per_question.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
    (args.out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
