# Final answer-generation implementation

This is the implementation loaded by `submission/solution.py`. It uses table-aware passages, BM25 retrieval, small generic source/qualifier adjustments, and evidence-constrained OpenAI answer generation. The evaluator calls `build()`, then `ingest()`, `answer()`, and `conflicts()` through the submission entry point.

There is no vector database or embedding path in this implementation. The separate embedding experiment was slower and did not improve the public end-to-end score, so its code and dependency were removed. The lexical retrieval and answer-generation fallback are the tested final path.

Set `OPENAI_API_KEY` in the environment to use `gpt-4.1-mini` for answer synthesis. For local development only, the ignored kit-root `.env` is also read. If the API is unavailable, the code falls back to retrieved evidence and marks the answer unconfident. Never commit an API key or the snapshot corpus.
