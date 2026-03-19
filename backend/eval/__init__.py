"""Eval package — golden dataset runner and LLM-as-judge scoring."""

from backend.eval.judge import (
    score_context_precision,
    score_correctness,
    score_faithfulness,
    score_hallucination,
)
from backend.eval.runner import run_eval

__all__ = [
    "run_eval",
    "score_correctness",
    "score_hallucination",
    "score_faithfulness",
    "score_context_precision",
]
