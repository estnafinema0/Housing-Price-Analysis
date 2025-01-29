import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge


class ExponentialLinearRegression(BaseEstimator, RegressorMixin):
    """A linear regression model that applies exponential transformation to targets.
    
    This model logarithmically transforms the target values before fitting,
    then exponentially transforms predictions back to the original scale.
    It uses Ridge regression internally for L2 regularization.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the model with Ridge regression parameters."""
        self.model = Ridge(*args, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ExponentialLinearRegression':
        if (y <= 0).any():
            raise ValueError("Target values must be positive for log transformation")
        
        self.model.fit(X, np.log(y))
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        log_predictions = self.model.predict(X)
        return np.exp(log_predictions)
    
    def get_params(self, deep: bool = True) -> dict:
        return self.model.get_params(deep=deep)
    
    def set_params(self, **params) -> 'ExponentialLinearRegression':
        self.model.set_params(**params)
        return self


class SGDLinearRegressor(BaseEstimator, RegressorMixin):
    """Custom Linear Regression using Stochastic Gradient Descent.
    
    This implementation includes L2 regularization and uses mini-batch SGD
    for optimization.
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        regularization: float = 1.0,
        delta_converged: float = 1e-3,
        max_steps: int = 1000,
        batch_size: int = 64
    ):
        self.lr = lr
        self.regularization = regularization
        self.delta_converged = delta_converged
        self.max_steps = max_steps
        self.batch_size = batch_size
        
        self.W = None
        self.b = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SGDLinearRegressor':
        n_features = X.shape[1]
        self.W = np.zeros(n_features)
        self.b = 0
        
        n_samples = X.shape[0]
        prev_W = np.zeros_like(self.W)
        prev_b = 0
        
        for step in range(self.max_steps):
            batch_indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            y_pred = X_batch @ self.W + self.b
            
            grad_W = (2/self.batch_size) * X_batch.T @ (y_pred - y_batch) + \
                    2 * self.regularization * self.W
            grad_b = (2/self.batch_size) * np.sum(y_pred - y_batch)
            
            self.W = self.W - self.lr * grad_W
            self.b = self.b - self.lr * grad_b
            
            W_diff = np.linalg.norm(self.W - prev_W)
            b_diff = abs(self.b - prev_b)
            if W_diff < self.delta_converged and b_diff < self.delta_converged:
                break
                
            prev_W = self.W.copy()
            prev_b = self.b
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b


def root_mean_squared_logarithmic_error(y_true: np.ndarray, y_pred: np.ndarray, a_min: float = 1.0) -> float:
    if (y_true < 0).any():
        raise ValueError("y_true contains negative values")
    
    y_pred_clipped = np.maximum(y_pred, a_min)
    
    return np.sqrt(np.mean((np.log(y_true) - np.log(y_pred_clipped)) ** 2))
