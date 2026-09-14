"""Dataset generation tests."""

import pytest
from fastapi.testclient import TestClient

from neuronscope.datasets import DatasetConfig, DatasetKind, generate_dataset
from neuronscope.main import create_app


@pytest.mark.parametrize("kind", list(DatasetKind))
def test_each_dataset_is_balanced_and_has_requested_size(kind: DatasetKind) -> None:
    result = generate_dataset(DatasetConfig(kind=kind, samples=101, seed=7))

    assert len(result.points) == 101
    labels = [point.label for point in result.points]
    assert set(labels) == {0, 1}
    assert abs(labels.count(0) - labels.count(1)) <= 1


def test_generation_is_reproducible_without_global_state() -> None:
    config = DatasetConfig(kind=DatasetKind.SPIRAL, samples=80, noise=0.08, seed=23)

    assert generate_dataset(config) == generate_dataset(config)
    assert generate_dataset(config) != generate_dataset(config.model_copy(update={"seed": 24}))


def test_dataset_endpoint_serializes_typed_points() -> None:
    response = TestClient(create_app()).post(
        "/api/v1/datasets/generate",
        json={"kind": "xor", "samples": 40, "noise": 0, "seed": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["config"]["kind"] == "xor"
    assert len(body["points"]) == 40
    assert set(body["points"][0]) == {"x", "y", "label"}


def test_dataset_endpoint_rejects_invalid_configuration() -> None:
    response = TestClient(create_app()).post(
        "/api/v1/datasets/generate", json={"kind": "unknown", "samples": 3}
    )

    assert response.status_code == 422
