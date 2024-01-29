from entity import Hyperparameters, Penguin
from penguin_predictor import train, predict
import json
import os
import pytest


TRAINING_JSON = os.path.join("test_data", "penguins_train.json")


@pytest.fixture
def training_data():
    penguins = [Penguin(**p) for p in json.load(open(TRAINING_JSON))]
    return penguins


def test_train_and_predict(training_data, tmp_path):
    tmp_path = str(tmp_path)
    hyp = Hyperparameters(
        target="species",
        training_size=0.8,
        regularization_strength=1.0,
    )
    metadata = train(data=training_data, hyperparameters=hyp, output_path=tmp_path)
    filename = os.path.join(tmp_path, f"model_{metadata.id}.pkl")
    assert os.path.exists(filename)

    # assert ...

    # a single penguin for testing
    pingu = Penguin(
        flipper_length_mm=250,
        bill_depth_mm=37,
        bill_length_mm=40,
        sex="Female",
    )
    response = predict(data=pingu, model_id=metadata.id, model_path=tmp_path)
    assert response.species == "Chinstrap"
    assert response.probability > 0.334
