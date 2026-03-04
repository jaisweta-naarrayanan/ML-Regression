import numpy as np
import pandas as pd

def load_data(filename):
    """
    Loads data from a CSV file and returns X and y.
    """
    df = pd.read_csv(filename)
    X = df['x'].values.reshape(-1, 1)
    y = df['y'].values.reshape(-1, 1)
    return X, y

def calculate_mse(X, y, theta):
    """
    Calculates the Mean Squared Error (MSE).
    MSE = 1/N * sum((y - prediction)^2)
    """
    N = len(y)
    prediction = X @ theta
    mse = (1/N) * np.sum((y - prediction) ** 2)
    return mse

def closed_form_solution(X, y):
    """
    Computes theta using the Normal Equation: theta = (X^T X)^-1 X^T y
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y

def gradient_descent(X, y, learning_rate, iterations):
    """
    Computes theta using Gradient Descent.
    Update rule: theta = theta - eta * 2 * X.T(X * theta - y)
    """
    theta = np.zeros((X.shape[1], 1))
    
    for i in range(iterations):
        gradient = 2 * X.T @ (X @ theta - y)
        theta = theta - learning_rate * gradient
        
        # Check for divergence
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            print(f"Gradient Descent diverged at iteration {i} with learning rate {learning_rate}")
            return theta
            
    return theta
