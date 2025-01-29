from typing import List, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from preprocessors import BaseDataPreprocessor, OneHotPreprocessor
from models import ExponentialLinearRegression


def get_continuous_columns() -> List[str]:
    """Get the list of continuous columns used in the housing price prediction."""
    return [
        'Lot_Frontage', 'Lot_Area', 'Year_Built', 'Year_Remod_Add',
        'Mas_Vnr_Area', 'BsmtFin_SF_1', 'BsmtFin_SF_2', 'Bsmt_Unf_SF',
        'Total_Bsmt_SF', 'First_Flr_SF', 'Second_Flr_SF', 'Low_Qual_Fin_SF',
        'Gr_Liv_Area', 'Bsmt_Full_Bath', 'Bsmt_Half_Bath', 'Full_Bath',
        'Half_Bath', 'Bedroom_AbvGr', 'Kitchen_AbvGr', 'TotRms_AbvGrd',
        'Fireplaces', 'Garage_Cars', 'Garage_Area', 'Wood_Deck_SF',
        'Open_Porch_SF', 'Enclosed_Porch', 'Three_season_porch',
        'Screen_Porch', 'Pool_Area', 'Misc_Val', 'Mo_Sold', 'Year_Sold',
        'Longitude', 'Latitude'
    ]


def get_categorical_columns() -> List[str]:
    """Get the list of categorical columns used in the housing price prediction."""
    return [
        "Overall_Qual",
        "Garage_Qual",
        "Sale_Condition",
        "MS_Zoning"
    ]


def make_base_pipeline(needed_columns: Optional[List[str]] = None, alpha: float = 1.0) -> Pipeline:
    """Create a basic pipeline with standard scaling and Ridge regression"""
    if needed_columns is None:
        needed_columns = get_continuous_columns()
        
    return Pipeline([
        ('preprocessor', BaseDataPreprocessor(needed_columns=needed_columns)),
        ('regressor', Ridge(alpha=alpha))
    ])


def make_onehot_pipeline(needed_columns: Optional[List[str]] = None, alpha: float = 1.0) -> Pipeline:
    """Pipeline with one-hot encoding and Ridge regression."""
    if needed_columns is None:
        needed_columns = get_continuous_columns()
        
    return Pipeline([
        ('preprocessor', OneHotPreprocessor(needed_columns=needed_columns)),
        ('regressor', Ridge(alpha=alpha))
    ])


def make_exponential_pipeline(needed_columns: Optional[List[str]] = None, alpha: float = 1.0) -> Pipeline:
    """Pipeline with one-hot encoding and exponential regression."""
    if needed_columns is None:
        needed_columns = get_continuous_columns()
        
    return Pipeline([
        ('preprocessor', OneHotPreprocessor(needed_columns=needed_columns)),
        ('regressor', ExponentialLinearRegression(alpha=alpha))
    ])


def make_ultimate_pipeline() -> Pipeline:
    """Pipeline with one-hot encoding and exponential regression."""
    return Pipeline([
        ('preprocessor', OneHotPreprocessor(needed_columns=get_continuous_columns())),
        ('regressor', ExponentialLinearRegression(alpha=1.0))
    ])
