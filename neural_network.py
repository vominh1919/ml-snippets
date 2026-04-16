#!/usr/bin/env python3
"""Simple neural network implementation."""

import numpy as np
from typing import List, Tuple

class NeuralNetwork:
    """Feedforward neural network."""
    
    def __init__(self, layers: List[int], learning_rate: float = 0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.01
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid."""
        return x * (1 - x)
    
    def forward(self, X: np.ndarray) -> List[np.ndarray]:
        """Forward pass."""
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)
        return activations
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray]) -> None:
        """Backward pass."""
        m = X.shape[0]
        
        # Calculate output layer error
        error = y - activations[-1]
        delta = error * self.sigmoid_derivative(activations[-1])
        
        # Update weights and biases
        for i in range(len(self.weights) - 1, -1, -1):
            self.weights[i] += self.learning_rate * np.dot(activations[i].T, delta) / m
            self.biases[i] += self.learning_rate * np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000) -> None:
        """Train the network."""
        for epoch in range(epochs):
            activations = self.forward(X)
            self.backward(X, y, activations)
            
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - activations[-1]))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(X)[-1]

# Example usage
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    nn = NeuralNetwork([2, 4, 1])
    nn.train(X, y, epochs=1000)
    predictions = nn.predict(X)
    print(f"Predictions: {predictions}")
