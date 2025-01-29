import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Optional, List, Dict, Union
from .column_definitions import get_continuous_columns, get_categorical_columns

class BaseDataPreprocessor(BaseEstimator, TransformerMixin):
    """Basic preprocessor that scales continuous features."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.continuous_cols = get_continuous_columns()
        
    def fit(self, X: pd.DataFrame, y=None):
        # Выбираем только числовые колонки
        X_numeric = X[self.continuous_cols]
        self.scaler.fit(X_numeric)
        return self
    
    def transform(self, X: pd.DataFrame):
        # Выбираем только числовые колонки
        X_numeric = X[self.continuous_cols]
        return self.scaler.transform(X_numeric)
    
    def fit_transform(self, X: pd.DataFrame, y=None):
        # Выбираем только числовые колонки
        X_numeric = X[self.continuous_cols]
        return self.scaler.fit_transform(X_numeric)

class OneHotPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor that applies one-hot encoding to categorical features 
    and scales continuous features."""
    
    def __init__(self):
        self.numeric_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.continuous_cols = get_continuous_columns()
        self.categorical_cols = get_categorical_columns()
        
    def fit(self, X: pd.DataFrame, y=None):
        # Обрабатываем числовые признаки
        X_numeric = X[self.continuous_cols]
        self.numeric_scaler.fit(X_numeric)
        
        # Обрабатываем категориальные признаки
        X_categorical = X[self.categorical_cols]
        self.categorical_encoder.fit(X_categorical)
        return self
    
    def transform(self, X: pd.DataFrame):
        # Преобразуем числовые признаки
        X_numeric = X[self.continuous_cols]
        X_numeric_scaled = self.numeric_scaler.transform(X_numeric)
        
        # Преобразуем категориальные признаки
        X_categorical = X[self.categorical_cols]
        X_categorical_encoded = self.categorical_encoder.transform(X_categorical)
        
        # Объединяем преобразованные признаки
        return np.hstack([X_numeric_scaled, X_categorical_encoded])
    
    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.fit(X).transform(X)

class SmartDataPreprocessor(BaseEstimator, TransformerMixin):
    """Advanced preprocessor with additional feature engineering."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.continuous_cols = get_continuous_columns()
        
    def fit(self, X: pd.DataFrame, y=None):
        X_numeric = X[self.continuous_cols]
        self.scaler.fit(X_numeric)
        return self
    
    def transform(self, X: pd.DataFrame):
        X_numeric = X[self.continuous_cols]
        X_scaled = self.scaler.transform(X_numeric)
        
        # Добавляем расчет расстояния до центра, если есть координаты
        if 'Longitude' in self.continuous_cols and 'Latitude' in self.continuous_cols:
            lon_idx = self.continuous_cols.index('Longitude')
            lat_idx = self.continuous_cols.index('Latitude')
            
            # Вычисляем среднее положение (центр)
            center_lon = np.mean(X_scaled[:, lon_idx])
            center_lat = np.mean(X_scaled[:, lat_idx])
            
            # Вычисляем расстояние до центра
            distances = np.sqrt(
                (X_scaled[:, lon_idx] - center_lon)**2 + 
                (X_scaled[:, lat_idx] - center_lat)**2
            )
            
            # Добавляем расстояние как новый признак
            X_scaled = np.column_stack([X_scaled, distances])
            
        return X_scaled