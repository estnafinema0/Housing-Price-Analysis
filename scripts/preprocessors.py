import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Optional, List, Dict, Union

class BaseDataPreprocessor(TransformerMixin):
    """Base class for data preprocessing with standardization capabilities.
    
    Attributes:
        needed_columns: List of column names to be processed.
        scaler: StandardScaler instance for feature scaling.
    """
    
    def __init__(self, needed_columns: Optional[List[str]] = None):
        self.needed_columns = needed_columns
        self.scaler = StandardScaler()

    def fit(self, data: pd.DataFrame, *args) -> 'BaseDataPreprocessor':
        if self.needed_columns is None:
            self.needed_columns = data.columns.tolist()
            
        selected_data = data[self.needed_columns]
        self.scaler.fit(selected_data)
        
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        selected_data = data[self.needed_columns]
        return self.scaler.transform(selected_data)


class SmartDataPreprocessor(TransformerMixin):
    """Advanced preprocessor with geographic feature engineering.
    
    Attributes:
        needed_columns: List of column names to be processed.
        scaler: StandardScaler instance for feature scaling.
        medians: Dictionary storing median values for imputation.
        city_center: Dictionary storing city center coordinates.
    """
    
    def __init__(self, needed_columns: Optional[List[str]] = None):
        self.needed_columns = needed_columns
        self.scaler = StandardScaler()
        self.medians = {}
        self.city_center = None
        
    def _calculate_city_center(self, data: pd.DataFrame) -> None:
        self.city_center = {
            'longitude': data['Longitude'].median(),
            'latitude': data['Latitude'].median()
        }
    
    def _calculate_distance_to_center(self, data: pd.DataFrame) -> np.ndarray:
        return np.sqrt(
            (data['Longitude'] - self.city_center['longitude']) ** 2 +
            (data['Latitude'] - self.city_center['latitude']) ** 2
        )
    
    def fit(self, data: pd.DataFrame, *args) -> 'SmartDataPreprocessor':
        if self.needed_columns is None:
            self.needed_columns = data.columns.tolist()
        
        self.medians['Lot_Frontage'] = data['Lot_Frontage'][data['Lot_Frontage'] > 0].median()
        
        self._calculate_city_center(data)
        transformed_data = self._transform_features(data)
        self.scaler.fit(transformed_data)
        
        return self
    
    def _transform_features(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        
        mask = result['Lot_Frontage'] == 0
        result.loc[mask, 'Lot_Frontage'] = self.medians['Lot_Frontage']
        
        result['Distance_To_Center'] = self._calculate_distance_to_center(result)
        
        if self.needed_columns is not None:
            selected_columns = self.needed_columns + ['Distance_To_Center']
            result = result[selected_columns]
            
        return result
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        transformed_data = self._transform_features(data)
        return self.scaler.transform(transformed_data)


class OneHotPreprocessor(BaseDataPreprocessor):    
    def __init__(self, categorical_columns: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = OneHotEncoder(
            drop='if_binary',
            handle_unknown='ignore',
            sparse_output=False
        )
        self.categorical_columns = categorical_columns or [
            "Overall_Qual",
            "Garage_Qual",
            "Sale_Condition",
            "MS_Zoning"
        ]

    def fit(self, data: pd.DataFrame, *args) -> 'OneHotPreprocessor':
        super().fit(data, *args)
        categorical_data = data[self.categorical_columns]
        self.encoder.fit(categorical_data)
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        numerical_features = super().transform(data)
        categorical_data = data[self.categorical_columns]
        categorical_features = self.encoder.transform(categorical_data)
        
        return np.hstack([numerical_features, categorical_features])