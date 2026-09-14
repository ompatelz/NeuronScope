"""Public deterministic diagnostics interface."""

from neuronscope.diagnostics.config import DiagnosticsConfig
from neuronscope.diagnostics.engine import evaluate_diagnostics
from neuronscope.diagnostics.schemas import (
    DiagnosticEvidence,
    DiagnosticResult,
    DiagnosticSeverity,
    DiagnosticType,
)

__all__ = [
    "DiagnosticEvidence",
    "DiagnosticResult",
    "DiagnosticSeverity",
    "DiagnosticType",
    "DiagnosticsConfig",
    "evaluate_diagnostics",
]
