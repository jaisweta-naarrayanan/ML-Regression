import numpy as np
import matplotlib.pyplot as plt
import regression_utils as utils

def create_quadratic_features(X):
    """
    Creates quadratic features [1, x, x^2].
    """
    return np.hstack((np.ones((X.shape[0], 1)), X, X**2))

def main():
    # File paths
    train_file = "q7-train.csv"
    test_file = "q7-test.csv"
    
    # 1. Load Data
    print("Loading data...")
    X_train_raw, y_train = utils.load_data(train_file)
    X_test_raw, y_test = utils.load_data(test_file)
    
    # 2. Add Quadratic Features
    X_train = create_quadratic_features(X_train_raw)
    X_test = create_quadratic_features(X_test_raw)
    
    # 3. Closed-Form Solution
    print("\n--- (a) Closed-Form Solution (Quadratic) ---")
    theta_closed = utils.closed_form_solution(X_train, y_train)
    mse_train_closed = utils.calculate_mse(X_train, y_train, theta_closed)
    mse_test_closed = utils.calculate_mse(X_test, y_test, theta_closed)
    
    print(f"Theta (Closed-Form):")
    print(f"  Theta0 (Intercept): {theta_closed[0][0]}")
    print(f"  Theta1 (Linear):    {theta_closed[1][0]}")
    print(f"  Theta2 (Quadratic): {theta_closed[2][0]}")
    print(f"MSE (Training):       {mse_train_closed}")
    print(f"MSE (Testing):        {mse_test_closed}")
    
    # 4. Gradient Descent
    print("\n--- (b) Gradient Descent (Quadratic) ---")
    # Quadratic features (x^2) can be large, so we need a smaller learning rate.

    learning_rate = 1e-7
    iterations = 500000 
    
    theta_gd = utils.gradient_descent(X_train, y_train, learning_rate, iterations)
    mse_train_gd = utils.calculate_mse(X_train, y_train, theta_gd)
    mse_test_gd = utils.calculate_mse(X_test, y_test, theta_gd)
    
    print(f"Parameters:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Iterations:    {iterations}")
    print(f"Theta (Gradient Descent):")
    print(f"  Theta0: {theta_gd[0][0]}")
    print(f"  Theta1: {theta_gd[1][0]}")
    print(f"  Theta2: {theta_gd[2][0]}")
    print(f"MSE (Training): {mse_train_gd}")
    print(f"MSE (Testing):  {mse_test_gd}")
    
    # 5. Model Comparison
    print("\n--- Comparison ---")
    theta_diff = np.abs(theta_closed - theta_gd)
    print("Theta Difference (Absolute):")
    print(theta_diff)
    
    # 6. Plotting
    plt.figure(figsize=(10, 6))
    
    # Scatter plots
    plt.scatter(X_train_raw, y_train, color='blue', label='Training Data', alpha=0.5)
    plt.scatter(X_test_raw, y_test, color='red', label='Testing Data', alpha=0.5)
    
    # Generate curve points
    x_range = np.linspace(X_train_raw.min() - 1, X_train_raw.max() + 1, 100).reshape(-1, 1)
    x_range_features = create_quadratic_features(x_range)
    
    y_closed = x_range_features @ theta_closed
    y_gd = x_range_features @ theta_gd
    
    # Plot curves
    plt.plot(x_range, y_closed, color='green', linestyle='--', linewidth=2, 
             label='Closed-Form Quadratic')
    plt.plot(x_range, y_gd, color='orange', linestyle=':', linewidth=3, 
             label='Gradient Descent Quadratic')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Quadratic Regression: Closed-Form vs Gradient Descent')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    output_plot = 'quadratic_regression_plot.png'
    plt.savefig(output_plot)
    print(f"\nPlot saved to {output_plot}")
    plt.close()

if __name__ == "__main__":
    main()
