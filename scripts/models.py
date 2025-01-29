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
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        if (y <= 0).any():
            raise ValueError("Target values must be positive for log transformation")
        
        self.model.fit(X, np.log(y))
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        log_predictions = self.model.predict(X)
        return np.exp(log_predictions)
    
    def get_params(self, deep: bool = True) -> dict:
        return self.model.get_params(deep=deep)
    
    def set_params(self, **params):
        self.model.set_params(**params)
        return self


class SGDLinearRegressor(BaseEstimator, RegressorMixin):
    """Custom Linear Regression using Stochastic Gradient Descent.
    
    This implementation includes L2 regularization and uses mini-batch SGD
    for optimization.
    """
    
    def __init__(self, learning_rate=0.01, batch_size=32, n_iterations=1000, 
                 l2_reg=0.0, momentum=0.9):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.l2_reg = l2_reg
        self.momentum = momentum
        
        # Добавляем списки для хранения истории
        self.loss_history = []
        self.weight_norm_history = []
        self.gradient_norm_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SGDLinearRegressor':
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.W = np.zeros(n_features)
        self.b = 0
        self.v_W = np.zeros_like(self.W)  # momentum for weights
        self.v_b = 0  # momentum for bias
        
        # Training loop
        for i in range(self.n_iterations):
            # Sample random batch
            batch_indices = np.random.choice(n_samples, self.batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Forward pass
            y_pred = X_batch @ self.W + self.b
            
            # Compute gradients
            grad_W = (2/self.batch_size) * X_batch.T @ (y_pred - y_batch) + \
                    2 * self.l2_reg * self.W
            grad_b = (2/self.batch_size) * np.sum(y_pred - y_batch)
            
            # Update with momentum
            self.v_W = self.momentum * self.v_W - self.learning_rate * grad_W
            self.v_b = self.momentum * self.v_b - self.learning_rate * grad_b
            
            self.W += self.v_W
            self.b += self.v_b
            
            # Save history
            loss = np.mean((y_pred - y_batch) ** 2) + self.l2_reg * np.sum(self.W ** 2)
            self.loss_history.append(loss)
            self.weight_norm_history.append(np.linalg.norm(self.W))
            self.gradient_norm_history.append(np.linalg.norm(grad_W))
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b


def root_mean_squared_logarithmic_error(y_true: np.ndarray, y_pred: np.ndarray, a_min: float = 1.0) -> float:
    if (y_true < 0).any():
        raise ValueError("y_true contains negative values")
    
    y_pred_clipped = np.maximum(y_pred, a_min)
    
    return np.sqrt(np.mean((np.log(y_true) - np.log(y_pred_clipped)) ** 2))
