"""Boundary tests for transparent diagnostic rules."""

import pytest

from neuronscope.diagnostics import DiagnosticsConfig, DiagnosticType, evaluate_diagnostics
from neuronscope.instrumentation import (
    ActivationStatistics,
    EpochInstrumentation,
    GradientStatistics,
    LayerInstrumentation,
)
from neuronscope.models import ActivationName


def _history(
    norms: tuple[float, ...], *, zeros: float = 0.0, nonfinite: int = 0
) -> tuple[EpochInstrumentation, ...]:
    return tuple(
        EpochInstrumentation(
            epoch=epoch,
            layers=(
                LayerInstrumentation(
                    layer_name="hidden_0",
                    activation=ActivationStatistics(
                        mean=0.1,
                        std=0.2,
                        minimum=0.0,
                        maximum=1.0,
                        zero_percentage=zeros,
                        nonfinite_count=0,
                    ),
                    weight_norm=1.0,
                    gradients=GradientStatistics(
                        norm=norm, mean=0.0, std=norm, nonfinite_count=nonfinite
                    ),
                ),
            ),
        )
        for epoch, norm in enumerate(norms, start=1)
    )


def test_vanishing_requires_consecutive_values_including_threshold() -> None:
    config = DiagnosticsConfig(consecutive_epochs=3, vanishing_gradient_norm=0.01)
    triggered = evaluate_diagnostics(_history((0.01, 0.009, 0.008)), config, ActivationName.TANH)
    interrupted = evaluate_diagnostics(_history((0.009, 0.02, 0.008)), config, ActivationName.TANH)
    assert [result.type for result in triggered] == [DiagnosticType.VANISHING_GRADIENTS]
    assert triggered[0].evidence[0].epochs == (1, 2, 3)
    assert interrupted == ()


def test_exploding_boundary_and_nonfinite_gradient() -> None:
    config = DiagnosticsConfig(consecutive_epochs=2, exploding_gradient_norm=10.0)
    finite = evaluate_diagnostics(_history((10.0, 11.0)), config, ActivationName.RELU)
    nonfinite = evaluate_diagnostics(_history((1.0, 1.0), nonfinite=1), config, ActivationName.RELU)
    assert finite[0].type is DiagnosticType.EXPLODING_GRADIENTS
    assert finite[0].severity == "warning"
    assert nonfinite[0].severity == "critical"
    assert nonfinite[0].evidence[0].metric == "gradient_nonfinite_count"


def test_dead_relu_requires_relu_and_consecutive_threshold() -> None:
    config = DiagnosticsConfig(consecutive_epochs=2, dead_relu_zero_percentage=90.0)
    relu = evaluate_diagnostics(_history((1.0, 1.0), zeros=90.0), config, ActivationName.RELU)
    tanh = evaluate_diagnostics(_history((1.0, 1.0), zeros=100.0), config, ActivationName.TANH)
    assert [result.type for result in relu] == [DiagnosticType.DEAD_RELU]
    assert tanh == ()


@pytest.mark.parametrize(
    "values",
    [
        {"consecutive_epochs": 1},
        {"vanishing_gradient_norm": 0},
        {"exploding_gradient_norm": 0},
        {"dead_relu_zero_percentage": 101},
    ],
)
def test_thresholds_are_validated(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DiagnosticsConfig.model_validate(values)
