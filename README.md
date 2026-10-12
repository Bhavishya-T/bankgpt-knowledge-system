# BankGPT knowledge-system submission

This project answers credit-union questions from five dated website snapshots. The challenge is to choose the right source, product tier, and snapshot date—and to say when the supplied pages do not establish an answer.

## What to submit

The scorer's entry folder is `submission/`. The required `solution.py` exposes `build()` and returns an object with `ingest(snapshot)`, `answer(question, as_of=None)`, and `conflicts()`.

The entry file delegates to the tested implementation in `experiments/rag_openai/solution.py`. That implementation uses the table-aware BM25 index in `experiments/bm25/solution.py` and generic source/qualifier signals in `experiments/hybrid/solution.py`. Include those implementation files, plus `submission/requirements.txt` and `submission/NOTES.md`, in the same repository. Read the actual answer pipeline in `experiments/rag_openai/solution.py`; `submission/solution.py` is just the contract entry point.

## How the solution works

1. Ingest each complete snapshot in date order. Pages are split into passages while keeping table headers and nearby product labels attached to values.
2. For an answer, select the newest complete snapshot on or before `as_of` (or the newest snapshot when no date is given).
3. BM25 ranks passages by lexical relevance. Small, generic source-type and qualifier signals help distinguish official product information and exact terms/model years; they do not contain answer values.
4. If `OPENAI_API_KEY` is available, the model synthesizes a concise answer from retrieved passages only and identifies evidence for each claim. The code checks cited passage IDs and numeric claims. Without the API, it returns evidence passages as an explicitly unconfident fallback.
5. `conflicts()` looks for changed numeric passages in similar sections of the same page and prefers the later snapshot. This is a heuristic; the public evaluation command does not include the private conflict gold file, so that method was not separately measured here.

No practice answers or expected values are hard-coded. The implementation only reads the supplied snapshots and does not call live websites.

## What we tried and what the scores mean

- The supplied starter scored 0.093 on the 40-question public set.
- The tested BM25 + source/qualifier ranking + evidence-constrained OpenAI generation reached QA 0.6488. Its earlier run scored 0.6352 overall.
- The embedding experiment scored 0.6319 overall, with 851.7-second ingestion versus about 5 seconds for the lexical path. Retrieval coverage improved slightly, but end-to-end answers did not, so embeddings and their dependency were removed.
- The final cleaned implementation's API-enabled public run scored 0.6310 overall: QA 0.6446, consistency 0.975, speed score 0.9061, 40 questions, zero runtime errors, 5.0-second ingestion. This is slightly below the earlier 0.6352 baseline. The cleanup also fixed a general chunking bug where short numeric fact lines were mistaken for headings; that made the provided conflict self-test pass (F1 1.0 on its one gold item), while the public QA score moved down slightly. See local `results/final-submission/report.json` for the full report.

Remaining weak categories are conflict resolution (1/3), stale traps (0.45), and list completeness (0.64). Treat this as a measured baseline, not perfect coverage. The provided synthetic conflict test passes its one item, but `conflicts()` reaches its 300-entry cap on the real corpus; its precision is unmeasured and over-reporting is a known risk.

## Run locally

Python 3.9+ is required (the organizers install in Python 3.11). From the kit root:

```bash
python3 -m pip install -r submission/requirements.txt
python3 harness/run_eval.py --submission submission --eval eval/public_eval.jsonl --out results/final-submission
```

Set `OPENAI_API_KEY` in the environment for the API-enabled run. A kit-root `.env` is read for local development and is git-ignored. Without a key, the fallback runs without an API but its answer accuracy is expected to be lower.

The scorer unpacks `snapshots_zip/` into `snapshots/` if needed. The assignment says not to redistribute the snapshot data; keep the corpus and `.env` out of Git. For local clean-clone testing, place the provided kit data next to the repository before running the command.

## Before submission

- Grant the organizers read access to the GitHub repository.
- Make sure the required submission and implementation files listed above are included.
- Do not commit `.env`, API keys, result outputs, or snapshot data.
- Share the repository URL and the exact commit hash requested in the deadline email.
