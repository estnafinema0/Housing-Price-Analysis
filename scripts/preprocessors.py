import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing import Optional, List
from sklearn.preprocessing import OneHotEncoder

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
        transformed_data = self._transform_features(data)
        return self.scaler.transform(transformed_data)
    


class OneHotPreprocessor(BaseDataPreprocessor):
    def __init__(self, **kwargs):
        super(OneHotPreprocessor, self).__init__(**kwargs)
        self.encoder = OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False)
        self.categorical_columns = interesting_columns

    def fit(self, data, *args):
        super().fit(data, *args)
        categorical_data = data[self.categorical_columns]
        self.encoder.fit(categorical_data) 
        return self

    def transform(self, data):
        numerical_features = super().transform(data)

        categorical_data = data[self.categorical_columns]
        categorical_features = self.encoder.transform(categorical_data)
        return np.hstack([numerical_features, categorical_features])