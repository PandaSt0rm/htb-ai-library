"""
Backwards compatible metrics module. Use :mod:`htb_ai_library.evaluation` instead.
"""

from .evaluation import (  # noqa: F401
    evaluate_attack_effectiveness,
    analyze_model_confidence,
    print_attack_summary,
)

__all__ = [
    "evaluate_attack_effectiveness",
    "analyze_model_confidence",
    "print_attack_summary",
]
