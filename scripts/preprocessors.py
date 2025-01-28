import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing import Optional, List

class BaseDataPreprocessor(TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]]=None):        
        self.needed_columns = needed_columns
        self.scaler = StandardScaler()

    def fit(self, data, *args):
        if self.needed_columns is None:
            self.needed_columns = data.columns.tolist()
            
        selected_data = data[self.needed_columns] 
        self.scaler.fit(selected_data)
        
        return self

    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Transforms features so that they can be fed into the regressors
        :param data: pd.DataFrame with all available columns
        :return: np.array with preprocessed features
        """
        selected_data = data[self.needed_columns]
        
        return self.scaler.transform(selected_data)
    

class SmartDataPreprocessor(TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]]=None):
        self.needed_columns = needed_columns
        self.scaler = StandardScaler()
        self.medians = {}
        self.city_center = None
        
    def _calculate_city_center(self, data):
        self.city_center = {
            'longitude': data['Longitude'].median(),
            'latitude': data['Latitude'].median()
        }
    
    def _calculate_distance_to_center(self, data):
        return np.sqrt(
            (data['Longitude'] - self.city_center['longitude'])**2 + 
            (data['Latitude'] - self.city_center['latitude'])**2
        )
    
    def fit(self, data: pd.DataFrame, *args):
        """
        Prepares the class for future transformations
        :param data: pd.DataFrame with all available columns
        :return: self
        """
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
    
    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Transforms features so that they can be fed into the regressors
        :param data: pd.DataFrame with all available columns
        :return: np.array with preprocessed features
        """
        transformed_data = self._transform_features(data)
        
        return self.scaler.transform(transformed_data)