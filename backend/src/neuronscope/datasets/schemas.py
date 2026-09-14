"""Typed configuration and output models for synthetic datasets."""

from enum import StrEnum

from pydantic import BaseModel, Field


class DatasetKind(StrEnum):
    """Supported binary-classification datasets."""

    TWO_MOONS = "two_moons"
    CIRCLES = "circles"
    XOR = "xor"
    SPIRAL = "spiral"


class DatasetConfig(BaseModel):
    """Configuration for a reproducible two-dimensional dataset."""

    kind: DatasetKind = DatasetKind.TWO_MOONS
    samples: int = Field(default=200, ge=40, le=2_000)
    noise: float = Field(default=0.12, ge=0.0, le=0.5)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)


class DatasetPoint(BaseModel):
    """A labeled point in two-dimensional feature space."""

    x: float
    y: float
    label: int = Field(ge=0, le=1)


class DatasetResult(BaseModel):
    """A generated dataset and the exact configuration that produced it."""

    config: DatasetConfig
    points: list[DatasetPoint]
