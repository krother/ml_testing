"""
Define data types for public interface
"""

from typing import Literal

from pydantic import BaseModel


class Hyperparameters(BaseModel):
    regularization_strength: float = 1.0
    training_size: float = 0.8
    target: str = "species"
    one_hot_cols: list[str] = ["sex"]
    feature_engineering_params: dict[str, str|float] = {} # more generic


class ModelEvaluation(BaseModel):
    training_accuracy: float
    validation_accuracy: float


class ModelMetadata(BaseModel):
    id: str
    timestamp: str
    python_version: str
    scikit_version: str
    git_reference: str  # hash of a specific commit
    model_type: str
    version: str  # if the model is versioned
    runtime: int
    data_ref: str  # could be a reference to the actual data

    hyperparameters: Hyperparameters
    evaluation: ModelEvaluation


class Penguin(BaseModel):
    flipper_length_mm: float
    bill_length_mm: float
    bill_depth_mm: float
    sex: Literal["Male", "Female"]
    species: Literal["Adelie", "Chinstrap", "Gentoo"] | None = None


class PredictionResponse(BaseModel):
    species: str
    probability: float
