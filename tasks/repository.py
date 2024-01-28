"""
Data Repository
"""

import seaborn as sns
from entity import Penguin


def get_training_data() -> list[Penguin]:
    """
    Simulates the process of getting training data from a database.
    We assume the data coming out of this function to be clean.
    """
    df = sns.load_dataset("penguins")
    df.dropna(inplace=True)
    return [Penguin(**p) for p in df.to_dict(orient="records")]
