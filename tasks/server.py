"""
The REST API server for inference
"""
from fastapi import FastAPI
from penguin_predictor import train, predict
from entity import ModelMetadata, Hyperparameters, Penguin
import repository


app = FastAPI()

config = {
    "model_id": "f2cfdd1a-be1e-11ee-9a58-34c93da04ff6",
    "model_path": "models/", 
}

@app.post("/train")  # URL suffix, URL path or endpoint
def train_model(hyperparameters: Hyperparameters) -> ModelMetadata:
    data = repository.get_training_data()
    metadata = train(data=data, hyperparameters=hyperparameters, output_path=config["model_path"])
    return metadata


@app.post("/predict")
def predict_penguin(penguin: Penguin) -> PredictionResponse:
    """uses the ML model to make a prediction for a single data point"""
    ...
