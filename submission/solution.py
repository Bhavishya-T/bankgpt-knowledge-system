"""Required entry point for the tested evidence-grounded RAG implementation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = next(
    (parent for parent in Path(__file__).resolve().parents
     if (parent / "experiments" / "rag_openai" / "solution.py").exists()),
    None,
)
if ROOT is None:
    raise RuntimeError("Include experiments/rag_openai/solution.py with this submission.")

implementation = ROOT / "experiments" / "rag_openai" / "solution.py"
spec = importlib.util.spec_from_file_location("submitted_rag_implementation", implementation)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load implementation at {implementation}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def build():
    return module.build()
