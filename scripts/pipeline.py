from typing import List, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from .preprocessors import BaseDataPreprocessor, OneHotPreprocessor
from .models import ExponentialLinearRegression
from .column_definitions import get_continuous_columns, get_categorical_columns


def make_base_pipeline():
    return Pipeline([
        ('preprocessor', BaseDataPreprocessor()),
        ('regressor', Ridge())
    ])


def make_onehot_pipeline():
    return Pipeline([
        ('preprocessor', OneHotPreprocessor()),
        ('regressor', Ridge())
    ])


def make_ultimate_pipeline() -> Pipeline:
    """Pipeline with one-hot encoding and exponential regression."""
    return Pipeline([
        ('preprocessor', OneHotPreprocessor()),
        ('regressor', ExponentialLinearRegression())
    ])
