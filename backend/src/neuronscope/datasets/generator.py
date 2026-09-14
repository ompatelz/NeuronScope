"""Synthetic binary-classification dataset generation."""

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_circles, make_moons  # type: ignore[import-untyped]

from neuronscope.datasets.schemas import DatasetConfig, DatasetKind, DatasetPoint, DatasetResult


def _xor(config: DatasetConfig) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(config.seed)
    class_sizes = (config.samples // 2, config.samples - config.samples // 2)
    feature_parts: list[NDArray[np.float64]] = []
    label_parts: list[NDArray[np.int64]] = []
    for label, size in enumerate(class_sizes):
        magnitudes = rng.uniform(0.08, 1.0, size=(size, 2))
        x_signs = rng.choice(np.array([-1.0, 1.0]), size=size)
        y_signs = x_signs if label == 0 else -x_signs
        feature_parts.append(
            np.column_stack((magnitudes[:, 0] * x_signs, magnitudes[:, 1] * y_signs))
        )
        label_parts.append(np.full(size, label, dtype=np.int64))
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    if config.noise:
        features += rng.normal(0.0, config.noise, size=features.shape)
    order = rng.permutation(config.samples)
    return features[order], labels[order]


def _spiral(config: DatasetConfig) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(config.seed)
    class_sizes = (config.samples // 2, config.samples - config.samples // 2)
    feature_parts: list[NDArray[np.float64]] = []
    label_parts: list[NDArray[np.int64]] = []
    for label, size in enumerate(class_sizes):
        radius = np.sqrt(rng.uniform(0.02, 1.0, size=size))
        base_angle = radius * 2.25 * np.pi + label * np.pi
        angle = base_angle + rng.normal(0.0, config.noise * 1.5, size=size)
        feature_parts.append(np.column_stack((radius * np.cos(angle), radius * np.sin(angle))))
        label_parts.append(np.full(size, label, dtype=np.int64))
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    order = rng.permutation(config.samples)
    return features[order], labels[order]


def _generate_arrays(config: DatasetConfig) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if config.kind == DatasetKind.TWO_MOONS:
        features, labels = make_moons(
            n_samples=config.samples, noise=config.noise, random_state=config.seed
        )
    elif config.kind == DatasetKind.CIRCLES:
        features, labels = make_circles(
            n_samples=config.samples,
            noise=config.noise,
            factor=0.5,
            random_state=config.seed,
        )
    elif config.kind == DatasetKind.XOR:
        return _xor(config)
    else:
        return _spiral(config)
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def generate_dataset(config: DatasetConfig) -> DatasetResult:
    """Generate a deterministic dataset without mutating global random state."""

    features, labels = _generate_arrays(config)
    points = [
        DatasetPoint(x=float(row[0]), y=float(row[1]), label=int(label))
        for row, label in zip(features, labels, strict=True)
    ]
    return DatasetResult(config=config, points=points)
