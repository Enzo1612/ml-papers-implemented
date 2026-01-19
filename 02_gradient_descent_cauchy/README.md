# Gradient Descent (Cauchy, 1847)

This project implements the **Method of Steepest Descent**, first described by Augustin-Louis Cauchy. It serves as an iterative alternative to the analytical Normal Equation (Legendre) for solving Linear regression problems.

## Objective

Given a system $Ax = b$, finding the exact solution via $(A^T A)^{-1}$ is computationnaly expensive ($0(n^3)$). Gradient Descent approximates the solution by iteratively minimizing the error function $J(\theta)$ using first-order derivatives.

## How it works

### 1. The Cost Function (MSE)

We define the error surface as convex:
$$J(\theta) = \frac{1}{2m} \lVert A\theta - b \rVert^2$$

### 2. The Gradient

The direction of steepest ascent is the partial derivative with respect to $\theta$.

$$\nabla J(\theta) = \frac{1}{m} A^T (A\theta - b)$$

### 3. The Update Rule

We move in the opposite direction of the gradient:

$$\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)$$
_where $\alpha$ is the learning rate._

## Implementation Details

- **Initialization:** Zero vector or random small values.
- **Stopping Condition:** Fixed iterations or when $\Delta J < \epsilon$.
- **Vectorization:** Full matrix operations are used, avoiding Python loops over data points.

## Expected Results

For the dataset `(1,2), (2,4), (3,5)`:

- **Analytical (Legendre):** Intercept $\approx 0.67$, Slope $=1.5$.
- **Gradient Descent (Cauchy):** Converges to the same values given sufficient iterations.

## Complexity

### Computational complexity

Gradient descent is $O(knd)$ where $k$ is iterations, $n$ is the number of data samples and $d$ is the number of features.

### Spatial complexity

In order to store the transpose matrix, spatial complexity is $O(mn)$ where $m$ is the number of samples, $n$ the number of features.

## Results Comparison

| Method                    | Intercept | Slope | Notes                   |
| ------------------------- | --------- | ----- | ----------------------- |
| Least Squares (Legendre)  | 0.667     | 1.5   | Exact solution          |
| Gradient Descent (Cauchy) | 0.667     | 1.500 | 10k iterations, lr=0.01 |

## Reference

**Cauchy, A. L.** (1847). _Méthode générale pour la résolution des systèmes d'équations simultanées_. Comptes Rendus de l'Académie des Sciences.
