"""Deterministic rules over recent scalar instrumentation."""

from collections.abc import Callable

from neuronscope.diagnostics.config import DiagnosticsConfig
from neuronscope.diagnostics.schemas import (
    DiagnosticEvidence,
    DiagnosticResult,
    DiagnosticSeverity,
    DiagnosticType,
)
from neuronscope.instrumentation import EpochInstrumentation, LayerInstrumentation
from neuronscope.models import ActivationName


def _observations(
    history: tuple[EpochInstrumentation, ...], layer_name: str, count: int
) -> list[tuple[int, LayerInstrumentation]]:
    recent = sorted(history, key=lambda item: item.epoch)[-count:]
    result: list[tuple[int, LayerInstrumentation]] = []
    for epoch in recent:
        layer = next((item for item in epoch.layers if item.layer_name == layer_name), None)
        if layer is not None:
            result.append((epoch.epoch, layer))
    return result


def _evidence(
    observations: list[tuple[int, LayerInstrumentation]],
    required: int,
    metric: str,
    threshold: float,
    value: Callable[[LayerInstrumentation], float | None],
    triggered: Callable[[float], bool],
) -> DiagnosticEvidence | None:
    if len(observations) != required:
        return None
    values = [value(layer) for _, layer in observations]
    if any(item is None or not triggered(item) for item in values):
        return None
    return DiagnosticEvidence(
        layer_name=observations[0][1].layer_name,
        epochs=tuple(epoch for epoch, _ in observations),
        metric=metric,
        observed_values=tuple(float(item) for item in values if item is not None),
        threshold=threshold,
    )


def evaluate_diagnostics(
    history: tuple[EpochInstrumentation, ...],
    config: DiagnosticsConfig,
    activation: ActivationName,
) -> tuple[DiagnosticResult, ...]:
    """Apply transparent rules to the configured number of recent epochs."""

    if not history:
        return ()
    results: list[DiagnosticResult] = []
    for layer_name in (layer.layer_name for layer in history[-1].layers):
        observed = _observations(history, layer_name, config.consecutive_epochs)
        nonfinite = [
            (epoch, layer.gradients.nonfinite_count)
            for epoch, layer in observed
            if layer.gradients.nonfinite_count > 0
        ]
        if nonfinite:
            results.append(
                DiagnosticResult(
                    type=DiagnosticType.EXPLODING_GRADIENTS,
                    severity=DiagnosticSeverity.CRITICAL,
                    evidence=(
                        DiagnosticEvidence(
                            layer_name=layer_name,
                            epochs=tuple(epoch for epoch, _ in nonfinite),
                            metric="gradient_nonfinite_count",
                            observed_values=tuple(float(value) for _, value in nonfinite),
                            threshold=0.0,
                        ),
                    ),
                    explanation="One or more parameter gradients are non-finite.",
                    possible_actions=(
                        "Lower the learning rate.",
                        "Check inputs and loss calculations.",
                        "Investigate before applying gradient clipping.",
                    ),
                )
            )
            continue

        exploding = _evidence(
            observed,
            config.consecutive_epochs,
            "gradient_norm",
            config.exploding_gradient_norm,
            lambda layer: layer.gradients.norm,
            lambda value: value >= config.exploding_gradient_norm,
        )
        if exploding:
            results.append(
                DiagnosticResult(
                    type=DiagnosticType.EXPLODING_GRADIENTS,
                    severity=DiagnosticSeverity.WARNING,
                    evidence=(exploding,),
                    explanation="Gradient norms stayed above the configured threshold.",
                    possible_actions=(
                        "Lower the learning rate.",
                        "Try a suitable initialization.",
                        "Inspect whether the network is too deep.",
                    ),
                )
            )

        if layer_name == "output":
            continue
        vanishing = _evidence(
            observed,
            config.consecutive_epochs,
            "gradient_norm",
            config.vanishing_gradient_norm,
            lambda layer: layer.gradients.norm,
            lambda value: value <= config.vanishing_gradient_norm,
        )
        if vanishing:
            results.append(
                DiagnosticResult(
                    type=DiagnosticType.VANISHING_GRADIENTS,
                    severity=DiagnosticSeverity.WARNING,
                    evidence=(vanishing,),
                    explanation="A hidden layer received persistently tiny gradients.",
                    possible_actions=(
                        "Try ReLU or Tanh.",
                        "Reduce network depth.",
                        "Try activation-appropriate initialization.",
                    ),
                )
            )

        dead_relu = (
            _evidence(
                observed,
                config.consecutive_epochs,
                "activation_zero_percentage",
                config.dead_relu_zero_percentage,
                lambda layer: layer.activation.zero_percentage if layer.activation else None,
                lambda value: value >= config.dead_relu_zero_percentage,
            )
            if activation is ActivationName.RELU
            else None
        )
        if dead_relu:
            results.append(
                DiagnosticResult(
                    type=DiagnosticType.DEAD_RELU,
                    severity=DiagnosticSeverity.WARNING,
                    evidence=(dead_relu,),
                    explanation="A ReLU layer produced mostly zero activations repeatedly.",
                    possible_actions=(
                        "Lower the learning rate.",
                        "Try He initialization.",
                        "Reduce layer width or depth.",
                    ),
                )
            )
    return tuple(results)
