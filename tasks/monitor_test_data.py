"""
Monitor test data for multiple trained models
"""
import os
import json
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score

from entity import Penguin

MODEL_PATH = "models/"
TEST_JSON = "../data/penguins_test.json"

# load test data
test_penguins = [Penguin(**p) for p in json.load(open(TEST_JSON))]
df_test = pd.DataFrame([p.model_dump() for p in test_penguins])  # TODO: could be nicer
del df_test["species"]

ytrue = [p.species for p in test_penguins]

for fn in os.listdir("models"):
    if fn.startswith('model'):
        with open(f"{MODEL_PATH}/{fn}", "rb") as f:
            pipeline = pickle.load(f)
        meta_fn = fn.replace("model", "metadata")
        with open(f"{MODEL_PATH}/{meta_fn}", "rb") as f:
            metadata = pickle.load(f)
        
        ypred = pipeline.predict(df_test)
        acc = round(accuracy_score(y_pred=ypred, y_true=ytrue), 4)
        print(metadata.id[:8], "\t", metadata.timestamp, "\t", acc)
