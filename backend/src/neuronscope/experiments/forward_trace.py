"""Exact, bounded forward-pass traces for one generated dataset sample."""

import math
from collections.abc import Iterable
from typing import cast

import torch
from torch import Tensor

from neuronscope.datasets import DatasetResult
from neuronscope.experiments.schemas import ForwardLayerTrace, ForwardPassTrace
from neuronscope.models import ConfigurableMLP


def _json_values(values: Tensor) -> tuple[float | None, ...]:
    """Detach a one-sample tensor and replace non-finite values explicitly."""

    flattened: Iterable[float] = values.detach().cpu().reshape(-1).tolist()
    return tuple(value if math.isfinite(value) else None for value in flattened)


def compute_forward_trace(
    model: ConfigurableMLP,
    dataset: DatasetResult,
    sample_index: int,
) -> ForwardPassTrace:
    """Observe one real sample as it passes through the current model state."""

    point = dataset.points[sample_index]
    inputs = torch.tensor([[point.x, point.y]], dtype=torch.float32)
    was_training = model.training
    layers: list[ForwardLayerTrace] = []
    try:
        model.eval()
        with torch.no_grad():
            values = inputs
            for index, (linear, activation) in enumerate(
                zip(model.hidden_linears, model.hidden_activations, strict=True)
            ):
                pre_activations = cast(Tensor, linear(values))
                values = cast(Tensor, activation(pre_activations))
                layers.append(
                    ForwardLayerTrace(
                        layer_name=f"hidden_{index}",
                        activation_name=model.config.activation.value,
                        pre_activations=_json_values(pre_activations),
                        activations=_json_values(values),
                    )
                )
            logits = cast(Tensor, model.output_layer(values))
            probabilities = torch.sigmoid(logits)
            probability = float(probabilities.reshape(-1)[0].item())
            layers.append(
                ForwardLayerTrace(
                    layer_name="output",
                    activation_name="sigmoid",
                    pre_activations=_json_values(logits),
                    activations=_json_values(probabilities),
                )
            )
    finally:
        model.train(was_training)

    raw_logit = float(logits.reshape(-1)[0].item())
    return ForwardPassTrace(
        sample_index=sample_index,
        input_values=(point.x, point.y),
        expected_label=point.label,
        layers=tuple(layers),
        output_logit=raw_logit if math.isfinite(raw_logit) else None,
        predicted_probability=probability,
        predicted_label=int(probability >= 0.5),
    )
