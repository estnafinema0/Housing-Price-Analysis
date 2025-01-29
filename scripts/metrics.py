import numpy as np
from sklearn.metrics import make_scorer

def root_mean_squared_logarithmic_error(y_true, y_pred, a_min=1.):
    if (y_true < 0).any():
        raise ValueError("y_true contains negative values")
    
    y_pred_clipped = np.maximum(y_pred, a_min)
    
    return np.sqrt(np.mean((np.log(y_true) - np.log(y_pred_clipped)) ** 2))

rmsle_scorer = make_scorer(root_mean_squared_logarithmic_error, greater_is_better=False)
