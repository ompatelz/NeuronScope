"""Dataset preview API."""

from fastapi import APIRouter

from neuronscope.datasets import DatasetConfig, DatasetResult, generate_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/generate", response_model=DatasetResult)
def create_dataset(config: DatasetConfig) -> DatasetResult:
    """Generate a deterministic dataset for preview or experimentation."""

    return generate_dataset(config)
