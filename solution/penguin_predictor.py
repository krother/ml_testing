"""
A predictor for penguin species
"""
import pickle
import sys
import time
import uuid

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from entity import (Hyperparameters, ModelEvaluation, ModelMetadata, Penguin,
                    PredictionResponse)


def create_pipeline(hyperparameters: Hyperparameters) -> Pipeline:
    """
    A complete definition of the model pipeline
    including feature engineering and other preprocessing.
    """
    coltrans = ColumnTransformer(
        [("one_hot_encode", OneHotEncoder(), hyperparameters.one_hot_cols)], remainder="passthrough"
    )

    model = LogisticRegression(C=hyperparameters.regularization_strength)

    return make_pipeline(
        coltrans,
        MinMaxScaler(),
        model,
    )


def evaluate_model(pipeline, Xtrain, ytrain, Xval, yval) -> ModelEvaluation:
    """evaluate the model"""
    ypred = pipeline.predict(Xtrain)
    train_acc = accuracy_score(y_pred=ypred, y_true=ytrain)

    ypred_val = pipeline.predict(Xval)
    val_acc = accuracy_score(y_pred=ypred_val, y_true=yval)

    return ModelEvaluation(
        training_accuracy=train_acc,
        validation_accuracy=val_acc,
    )


def train(
    data: list[Penguin], hyperparameters: Hyperparameters, output_path: str
) -> ModelMetadata:
    """Trains the model and returns metadata"""
    # convert the data to a vectorized data structure
    df = pd.DataFrame([p.model_dump() for p in data])
    y = df[hyperparameters.target]
    X = df.drop(hyperparameters.target, axis=1)

    Xtrain, Xval, ytrain, yval = train_test_split(
        X, y, train_size=hyperparameters.training_size, random_state=42
    )

    pipeline = create_pipeline(hyperparameters)
    pipeline.fit(Xtrain, ytrain)

    evaluation = evaluate_model(pipeline, Xtrain, ytrain, Xval, yval)

    # final training with all data
    pipeline.fit(X, y)

    # save the model
    model_id = str(uuid.uuid1())
    with open(f"{output_path}/model_{model_id}.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    # save metadata
    metadata = ModelMetadata(
        id=model_id,
        timestamp=str(time.asctime()),
        python_version=str(sys.version),
        scikit_version=str(sklearn.__version__),
        git_reference="foo",  # hash of a specific commit
        model_type="LogReg",
        version="1.2.3",  # if the model is versioned
        runtime=123,
        data_ref="s3.aws.com/xxxxx",  # could be a reference to the actual data
        hyperparameters=hyperparameters,
        evaluation=evaluation,
    )
    with open(f"{output_path}/metadata_{model_id}.pkl", "wb") as f:
        pickle.dump(metadata, f)

    return metadata


def predict(data: Penguin, model_id: str, model_path: str) -> PredictionResponse:
    """produces a prediction for a single penguin"""
    with open(f"{model_path}/model_{model_id}.pkl", "rb") as f:
        df = pd.DataFrame([data.model_dump()])
        pipeline = pickle.load(f)
        return PredictionResponse(
            species=pipeline.predict(df)[0],
            probability=max(pipeline.predict_proba(df)[0]),
        )
