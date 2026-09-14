"""Removable PyTorch hooks and immediate tensor-to-scalar reduction."""

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from neuronscope.instrumentation.schemas import (
    ActivationStatistics,
    EpochInstrumentation,
    GradientStatistics,
    LayerInstrumentation,
)
from neuronscope.models import ConfigurableMLP


def summarize_activations(values: Tensor) -> ActivationStatistics:
    """Detach activation values and return only JSON-safe scalar statistics."""

    detached = values.detach()
    total = detached.numel()
    if total == 0:
        return ActivationStatistics(
            mean=None,
            std=None,
            minimum=None,
            maximum=None,
            zero_percentage=0.0,
            nonfinite_count=0,
        )

    finite_mask = torch.isfinite(detached)
    finite_values = detached[finite_mask].to(dtype=torch.float64)
    nonfinite_count = total - finite_values.numel()
    zero_percentage = float((detached == 0).sum().item()) * 100.0 / total
    if finite_values.numel() == 0:
        return ActivationStatistics(
            mean=None,
            std=None,
            minimum=None,
            maximum=None,
            zero_percentage=zero_percentage,
            nonfinite_count=nonfinite_count,
        )

    return ActivationStatistics(
        mean=float(finite_values.mean().item()),
        std=float(finite_values.std(unbiased=False).item()),
        minimum=float(finite_values.min().item()),
        maximum=float(finite_values.max().item()),
        zero_percentage=zero_percentage,
        nonfinite_count=nonfinite_count,
    )


def _summarize_gradients(parameters: Iterable[nn.Parameter]) -> GradientStatistics:
    finite_count = 0
    nonfinite_count = 0
    value_sum = 0.0
    square_sum = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite_mask = torch.isfinite(gradient)
        finite_values = gradient[finite_mask].to(dtype=torch.float64)
        nonfinite_count += gradient.numel() - finite_values.numel()
        finite_count += finite_values.numel()
        value_sum += float(finite_values.sum().item())
        square_sum += float(finite_values.square().sum().item())

    if finite_count == 0:
        return GradientStatistics(
            norm=None,
            mean=None,
            std=None,
            nonfinite_count=nonfinite_count,
        )

    mean = value_sum / finite_count
    variance = max(0.0, square_sum / finite_count - mean * mean)
    return GradientStatistics(
        norm=math.sqrt(square_sum),
        mean=mean,
        std=math.sqrt(variance),
        nonfinite_count=nonfinite_count,
    )


def _weight_norm(layer: nn.Linear) -> float | None:
    weight = layer.weight.detach()
    if not torch.isfinite(weight).all():
        return None
    return float(torch.linalg.vector_norm(weight.to(dtype=torch.float64)).item())


class TrainingInstrumentationCollector:
    """Capture one training pass at a time without retaining tensor references."""

    def __init__(self, model: ConfigurableMLP) -> None:
        self._model = model
        self._handles: list[RemovableHandle] = []
        self._active = False
        self._activations: dict[str, ActivationStatistics] = {}

    @property
    def attached(self) -> bool:
        """Report whether this collector currently owns registered hooks."""

        return bool(self._handles)

    def attach(self) -> None:
        """Register one hook per hidden activation; repeated calls are harmless."""

        if self.attached:
            return
        for index, module in enumerate(self._model.hidden_activations):
            layer_name = f"hidden_{index}"

            def capture(
                _module: nn.Module,
                _inputs: tuple[object, ...],
                output: object,
                *,
                name: str = layer_name,
            ) -> None:
                if self._active and isinstance(output, Tensor):
                    self._activations[name] = summarize_activations(output)

            self._handles.append(module.register_forward_hook(capture))

    def begin_training_pass(self) -> None:
        """Enable hooks for the next training forward pass only."""

        if not self.attached:
            raise RuntimeError("Instrumentation hooks must be attached before capture.")
        self._activations.clear()
        self._active = True

    def end_training_forward(self) -> None:
        """Disable activation capture before validation or metrics inference."""

        self._active = False

    def capture_epoch(self, epoch: int) -> EpochInstrumentation:
        """Read gradients after backward and return detached scalar summaries."""

        self._active = False
        learned_layers = [
            (f"hidden_{index}", layer)
            for index, layer in enumerate(self._model.hidden_linears)
            if isinstance(layer, nn.Linear)
        ]
        learned_layers.append(("output", self._model.output_layer))
        return EpochInstrumentation(
            epoch=epoch,
            layers=tuple(
                LayerInstrumentation(
                    layer_name=name,
                    activation=self._activations.get(name),
                    weight_norm=_weight_norm(layer),
                    gradients=_summarize_gradients(layer.parameters()),
                )
                for name, layer in learned_layers
            ),
        )

    def remove(self) -> None:
        """Remove every hook and discard any captured summaries."""

        self._active = False
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._activations.clear()
