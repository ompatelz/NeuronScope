"""Stable request and response contracts for synchronous experiments."""

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from neuronscope.datasets import DatasetConfig, DatasetResult
from neuronscope.models import MLPArchitecture, MLPConfig
from neuronscope.training import TrainingConfig, TrainingResult


class ExperimentRequest(BaseModel):
    """Validated configuration for one generated binary-classification run."""

    model_config = ConfigDict(frozen=True)

    dataset: DatasetConfig
    model: MLPConfig
    training: TrainingConfig

    @model_validator(mode="after")
    def validate_generated_dataset_shape(self) -> Self:
        """Require the model shape used by NeuronScope's generated 2D datasets."""

        if self.model.input_size != 2 or self.model.output_size != 1:
            raise ValueError(
                "Generated binary datasets require a model with input_size=2 and output_size=1."
            )
        return self


class ExperimentResponse(BaseModel):
    """Dataset, inspectable model structure, and observed training result."""

    model_config = ConfigDict(frozen=True)

    dataset: DatasetResult
    architecture: MLPArchitecture
    training: TrainingResult
