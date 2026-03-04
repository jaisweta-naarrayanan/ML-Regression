# ML Regression — CS6140 Machine Learning

Implementation of Linear, Quadratic, and Cubic Regression using two methods: **Closed-Form Solution** (Normal Equation) and **Gradient Descent** — from scratch using NumPy.

---

## Project Structure

```
ML-Regression/
├── regression_utils.py        # Shared utility functions (load_data, closed_form, gradient_descent, MSE)
├── linear_regression.py       # Part 1 — Linear model: y = θ₀ + θ₁x
├── quadratic_regression.py    # Part 2 — Quadratic model: y = θ₀ + θ₁x + θ₂x²
├── cubic_regression.py        # Part 3 — Cubic model: y = θ₀ + θ₁x + θ₂x² + θ₃x³
├── q7-train.csv               # Training dataset
├── q7-test.csv                # Testing dataset
├── linear_regression_plot.png
├── quadratic_regression_plot.png
└── cubic_regression_plot.png
```

---

## Assignment Overview

### Part 1 — Linear Regression (`linear_regression.py`)

Fits a linear model: **y = θ₀ + θ₁x**

- Loads `x` and `y` from CSV files
- Plots training points (blue) and testing points (red) using `plt.scatter`
- **(a) Closed-Form Solution** — uses the Normal Equation:  
  `θ = (XᵀX)⁻¹ Xᵀy`
- **(b) Gradient Descent** — iteratively updates θ using:  
  `θ ← θ − η · 2Xᵀ(Xθ − y)`  
  with `lr = 1e-6`, `iterations = 200,000`
- Reports θ₀, θ₁, training MSE, and testing MSE for both methods
- Plots both regression lines and compares results

---

### Part 2 — Quadratic Regression (`quadratic_regression.py`)

Fits a quadratic model: **y = θ₀ + θ₁x + θ₂x²**

- Constructs feature matrix `[1, x, x²]`
- **(a) Closed-Form Solution** — reports θ₀, θ₁, θ₂ and MSEs
- **(b) Gradient Descent** — `lr = 1e-7`, `iterations = 500,000`
- **(c) Comparison** — plots both fitted curves against training/testing scatter, reports absolute θ differences

---

### Part 3 — Cubic Regression (`cubic_regression.py`)

Fits a cubic model: **y = θ₀ + θ₁x + θ₂x² + θ₃x³**

- Constructs feature matrix `[1, x, x², x³]`
- **(a) Closed-Form Solution** — reports θ₀–θ₃ and MSEs
- **(b) Gradient Descent** — `lr = 1e-8`, `iterations = 1,000,000` (very small lr required due to x³ scale)
- **(c) Comparison** — plots both fitted curves, reports theta differences

---

## Shared Utilities (`regression_utils.py`)

| Function | Description |
|---|---|
| `load_data(filename)` | Reads CSV, returns `X` and `y` as NumPy arrays |
| `closed_form_solution(X, y)` | Normal Equation: `θ = (XᵀX)⁻¹Xᵀy` |
| `gradient_descent(X, y, lr, iters)` | Gradient descent with divergence check |
| `calculate_mse(X, y, theta)` | `MSE = (1/N) Σ(y − Xθ)²` |

---

## Setup & Usage

**Requirements:** Python 3.x with `numpy`, `pandas`, `matplotlib`

```bash
# Install dependencies
pip install numpy pandas matplotlib

# Run each model
python linear_regression.py
python quadratic_regression.py
python cubic_regression.py
```

Each script prints θ values and MSEs for both methods, and saves a plot as a `.png` file.

---

## Key Notes

- **Closed-form** gives the exact analytical solution but requires matrix inversion — `O(n³)` complexity.
- **Gradient Descent** is iterative and scales better but requires careful tuning of the learning rate, especially for higher-degree polynomials where feature values grow large.
- Learning rates decrease across models (`1e-6` → `1e-7` → `1e-8`) to prevent divergence as polynomial degree increases.

---
