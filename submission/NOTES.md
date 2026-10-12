# Submission notes

The entry point is `solution.py`; the answer-generation implementation is in `experiments/rag_openai/solution.py`, with passage chunking/BM25 and generic source/qualifier signals in `experiments/bm25/` and `experiments/hybrid/`. It ingests complete dated snapshots and chooses the latest snapshot available at the requested `as_of` date. The answer model sees only retrieved passages, identifies the requested product/tier/date, and returns evidence IDs for claims. The code validates those IDs and numeric claims. It does not use question-specific answers or call live websites.

The final implementation uses BM25 and an OpenAI model; there is no vector database or embedding code. We tested embeddings separately: they slightly improved retrieval coverage but moved the end-to-end score from 0.6352 to 0.6319 and increased ingestion from 4.8 to 851.7 seconds, so that path was removed.

Final public run after cleanup: 40 questions, QA 0.6446, consistency 0.975, speed score 0.9061, final score 0.6310, zero runtime errors, 5.0-second ingestion. The exact report is in the local `results/final-submission/report.json` (results are not part of the hand-in). The provided conflict smoke test matched its one gold contradiction (F1 1.0); on the real corpus, `conflicts()` reaches its 300-entry cap, so its precision is not established and over-reporting is a known risk.

`OPENAI_API_KEY` enables `gpt-4.1-mini` answer synthesis. Set it in the environment for the API-enabled benchmark. The ignored kit-root `.env` is read for local development; never commit or share it. Without a key, the deterministic fallback returns retrieved evidence as unconfident rather than inventing an answer. Install the dependency in `requirements.txt`.
