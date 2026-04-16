#!/usr/bin/env python3
"""Linear regression implementation."""

import numpy as np
from typing import Tuple

class LinearRegression:
    """Simple linear regression model."""
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Example usage
if __name__ == "__main__":
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    print(f"Predictions: {predictions}")
