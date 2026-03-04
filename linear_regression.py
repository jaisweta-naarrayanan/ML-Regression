import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename):
    """
    Loads data from a CSV file and returns X and y.
    """
    df = pd.read_csv(filename)
    X = df['x'].values.reshape(-1, 1)
    y = df['y'].values.reshape(-1, 1)
    return X, y

def add_bias(X):
    """
    Adds a column of ones to X for the intercept term.
    """
    return np.hstack((np.ones((X.shape[0], 1)), X))

def calculate_mse(X, y, theta):
    """
    Calculates the Mean Squared Error (MSE).
    MSE = 1/N * sum((y - (theta0 + theta1 * x))^2)
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

def gradient_descent(X, y, learning_rate=0.00001, iterations=50000):
    """
    Computes theta using Gradient Descent.
    Update rule: theta = theta - eta * 2 * X.T(X * theta - y)
    Using Sum of Squared Errors gradient as derived from l(theta) = ||X theta - y||^2
    """
    theta = np.zeros((X.shape[1], 1))
    
    for i in range(iterations):
        # Gradient of l(theta) = ||X theta - y||^2
        # grad = 2 * X.T @ (X @ theta - y)
        gradient = 2 * X.T @ (X @ theta - y)
        theta = theta - learning_rate * gradient
        
        if i % 10000 == 0:
            loss = np.sum((X @ theta - y) ** 2)
            # print(f"Iteration {i}: Loss {loss}")
            
    return theta

def main():
    # File paths
    train_file = "q7-train.csv"
    test_file = "q7-test.csv"
    
    # 1. Load and Preprocess Data
    print("Loading data...")
    X_train_raw, y_train = load_data(train_file)
    X_test_raw, y_test = load_data(test_file)
    
    X_train = add_bias(X_train_raw)
    X_test = add_bias(X_test_raw)
    
    # 2. Closed-Form Solution
    print("\n--- (a) Closed-Form Solution ---")
    theta_closed = closed_form_solution(X_train, y_train)
    mse_train_closed = calculate_mse(X_train, y_train, theta_closed)
    mse_test_closed = calculate_mse(X_test, y_test, theta_closed)
    
    print(f"Theta (Closed-Form):")
    print(f"  Theta0 (Intercept): {theta_closed[0][0]}")
    print(f"  Theta1 (Slope):     {theta_closed[1][0]}")
    print(f"MSE (Training):       {mse_train_closed}")
    print(f"MSE (Testing):        {mse_test_closed}")
    
    # 3. Gradient Descent
    print("\n--- (b) Gradient Descent ---")
    # Choosing learning rate and iterations
    # We need a very small learning rate.
    learning_rate = 1e-6 
    iterations = 200000 
    
    theta_gd = gradient_descent(X_train, y_train, learning_rate, iterations)
    mse_train_gd = calculate_mse(X_train, y_train, theta_gd)
    mse_test_gd = calculate_mse(X_test, y_test, theta_gd)
    
    print(f"Parameters:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Iterations:    {iterations}")
    print(f"Theta (Gradient Descent):")
    print(f"  Theta0 (Intercept): {theta_gd[0][0]}")
    print(f"  Theta1 (Slope):     {theta_gd[1][0]}")
    print(f"MSE (Training):       {mse_train_gd}")
    print(f"MSE (Testing):        {mse_test_gd}")
    
    # 4. Comparison and Discussion
    print("\n--- Comparison ---")
    theta_diff = np.abs(theta_closed - theta_gd)
    print(f"Absolute Difference in Theta:")
    print(f"  Theta0 Diff: {theta_diff[0][0]}")
    print(f"  Theta1 Diff: {theta_diff[1][0]}")
    
    print("\nDiscussion:")
    if np.all(theta_diff < 1e-3):
        print("The values are very close, indicating Gradient Descent converged successfully.")
    else:
        print("The values differ noticeably. This suggests Gradient Descent might need more iterations or a better learning rate.")
        
    mse_diff_train = abs(mse_train_closed - mse_train_gd)
    mse_diff_test = abs(mse_test_closed - mse_test_gd)
    print(f"Difference in MSE (Training): {mse_diff_train}")
    print(f"Difference in MSE (Testing):  {mse_diff_test}")

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    
    # Scatter plots
    plt.scatter(X_train_raw, y_train, color='blue', label='Training Data', alpha=0.5)
    plt.scatter(X_test_raw, y_test, color='red', label='Testing Data', alpha=0.5)
    
    # Generate line points
    x_range = np.linspace(X_train_raw.min() - 1, X_train_raw.max() + 1, 100).reshape(-1, 1)
    x_range_bias = add_bias(x_range)
    
    y_closed = x_range_bias @ theta_closed
    y_gd = x_range_bias @ theta_gd
    
    # Plot lines
    plt.plot(x_range, y_closed, color='green', linestyle='--', linewidth=2, 
             label=f'Closed-Form: y = {theta_closed[0][0]:.2f} + {theta_closed[1][0]:.2f}x')
    plt.plot(x_range, y_gd, color='orange', linestyle=':', linewidth=3, 
             label=f'Gradient Descent: y = {theta_gd[0][0]:.2f} + {theta_gd[1][0]:.2f}x')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression: Closed-Form vs Gradient Descent')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    output_plot = 'linear_regression_plot.png'
    plt.savefig(output_plot)
    print(f"\nPlot saved to {output_plot}")
    plt.close()

if __name__ == "__main__":
    main()
