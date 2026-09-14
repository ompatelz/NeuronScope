"""Synchronous full-batch training for small binary-classification experiments."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import SGD, Adam, Optimizer

from neuronscope.datasets import DatasetResult
from neuronscope.instrumentation import TrainingInstrumentationCollector
from neuronscope.instrumentation.schemas import EpochInstrumentation
from neuronscope.models import ConfigurableMLP
from neuronscope.training.schemas import (
    EpochMetrics,
    OptimizerName,
    TrainingConfig,
    TrainingResult,
)


@dataclass(frozen=True)
class TrainingRun:
    """Internal result whose temporary CPU states never cross the API boundary."""

    result: TrainingResult
    states: dict[int, dict[str, Tensor]]


def select_snapshot_epochs(epochs: int, max_snapshots: int) -> tuple[int, ...]:
    """Select first, last, and evenly spaced epochs within a strict cap."""

    count = min(epochs, max_snapshots)
    if count == 1:
        return (1,)
    return tuple(round(index * (epochs - 1) / (count - 1)) + 1 for index in range(count))


def _dataset_tensors(dataset: DatasetResult) -> tuple[Tensor, Tensor]:
    """Convert the transport-friendly point representation into training tensors."""

    features = torch.tensor([[point.x, point.y] for point in dataset.points], dtype=torch.float32)
    targets = torch.tensor([[float(point.label)] for point in dataset.points], dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise ValueError("Dataset features must all be finite.")
    return features, targets


def _optimizer(model: ConfigurableMLP, config: TrainingConfig) -> Optimizer:
    if config.optimizer is OptimizerName.SGD:
        return SGD(model.parameters(), lr=config.learning_rate)
    return Adam(model.parameters(), lr=config.learning_rate)


def _measure(logits: Tensor, targets: Tensor, loss_function: nn.Module) -> tuple[float, float]:
    loss = loss_function(logits, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Training produced a non-finite loss.")
    predictions = logits >= 0
    accuracy = (predictions == (targets >= 0.5)).float().mean()
    return float(loss.item()), float(accuracy.item())


def train_model(
    model: ConfigurableMLP,
    dataset: DatasetResult,
    config: TrainingConfig,
) -> TrainingResult:
    """Train a binary MLP on the CPU and return post-update metrics for every epoch.

    The engine intentionally uses the entire generated dataset as one batch. There is no
    shuffling or device-dependent execution, so equal dataset/model configurations produce
    repeatable runs while NeuronScope datasets remain small.
    """

    return train_model_with_snapshots(model, dataset, config, snapshot_epochs=()).result


def train_model_with_snapshots(
    model: ConfigurableMLP,
    dataset: DatasetResult,
    config: TrainingConfig,
    snapshot_epochs: tuple[int, ...],
) -> TrainingRun:
    """Train while temporarily retaining only explicitly selected CPU model states."""

    architecture = model.architecture
    if architecture.input_size != 2 or architecture.output_size != 1:
        raise ValueError(
            "Generated binary datasets require an MLP with input_size=2 and output_size=1."
        )
    if not dataset.points:
        raise ValueError("Training requires at least one dataset point.")

    features, targets = _dataset_tensors(dataset)
    model.to(torch.device("cpu"))
    model.train()
    optimizer = _optimizer(model, config)
    loss_function = nn.BCEWithLogitsLoss()
    history: list[EpochMetrics] = []
    instrumentation_history: list[EpochInstrumentation] = []
    captured_states: dict[int, dict[str, Tensor]] = {}
    selected_epochs = set(snapshot_epochs)
    collector = TrainingInstrumentationCollector(model) if config.instrumentation else None

    if collector is not None:
        collector.attach()
    try:
        for epoch in range(1, config.epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            if collector is not None:
                collector.begin_training_pass()
            logits = model(features)
            if collector is not None:
                collector.end_training_forward()
            loss = loss_function(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Training produced a non-finite loss at epoch {epoch}.")
            loss.backward()
            if collector is not None:
                instrumentation_history.append(collector.capture_epoch(epoch))
            optimizer.step()

            with torch.no_grad():
                loss_value, accuracy = _measure(model(features), targets, loss_function)
            history.append(EpochMetrics(epoch=epoch, loss=loss_value, accuracy=accuracy))
            if epoch in selected_epochs:
                captured_states[epoch] = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
    finally:
        if collector is not None:
            collector.remove()

    final = history[-1]
    return TrainingRun(
        result=TrainingResult(
            config=config,
            history=tuple(history),
            instrumentation=tuple(instrumentation_history),
            final_loss=final.loss,
            final_accuracy=final.accuracy,
        ),
        states=captured_states,
    )
