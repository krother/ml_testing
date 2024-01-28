"""
Define data types for public interface
"""
from pydantic import BaseModel
from typing import Literal


class Hyperparameters(BaseModel):
    regularization_strength: float = 1.0
    training_size: float = 0.8
    target: str = "species"


class ModelEvaluation(BaseModel):
    training_accuracy: float
    validation_accuracy: float


class ModelMetadata(BaseModel):
    ...
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
