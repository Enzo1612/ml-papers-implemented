# Legendre's Method of Least Squares (1805)

This project implements the **Method of Least Squares** from first principles, based on the appendix of _Nouvelles méthodes pour la détermination des orbites des comètes_ by A.-M. Legendre (1805).

It uses a custom Linear Algebra library (implemented from scratch in Python) to solve the **Normal Equation** via Gaussian Elimination with Partial Pivoting.

## 1. Mathematical Formulation

### The Objective

Given an overdetermined system $Ax = b$ where $A \in \mathbb{R}^{m \times n}$ ($m > n$), there is typically no exact solution. We seek the vector $x$ that minimizes the squared Euclidean norm of the residual:

$$\min_x \| Ax - b \|^2$$

### Derivation of the Normal Equation

Let $J(x) = \| Ax - b \|^2 = (Ax - b)^T (Ax - b)$.
Expanding this term:
$$J(x) = x^T A^T A x - 2x^T A^T b + b^T b$$

To find the minimum, we set the gradient with respect to $x$ to zero:
$$\nabla_x J(x) = 2 A^T A x - 2 A^T b = 0$$

Rearranging yields the **Normal Equation**:
$$A^T A x = A^T b$$

### Assumptions & Constraints

1.  **Full Rank:** The matrix $A$ must have full column rank (rank $n$). If $A$ is rank-deficient, $A^T A$ is singular (non-invertible), and the system has infinite solutions.
2.  **Linearity:** The relationship between inputs and parameters must be linear.
3.  **residuals:** The method assumes residuals are normally distributed (homoscedasticity) for the solution to be the Maximum Likelihood Estimator.

## 2. Algorithmic Complexity

The solver uses **Gaussian Elimination with Partial Pivoting**.

- **Time Complexity:** $O(n^3)$ for solving the system (where $n$ is the number of features/columns). Computing $A^T A$ takes $O(m n^2)$.
  - _Note:_ This scales poorly for large $n$. Modern iterative methods (Gradient Descent, Conjugate Gradient) are preferred for large-scale ML.
- **Space Complexity:** $O(n^2)$ to store the covariance matrix $A^T A$.

### Numerical Stability

- **Pivoting:** Partial pivoting is implemented to avoid division by zero and minimize rounding errors when the pivot element is small.
- **Conditioning:** If $A$ is ill-conditioned (columns are nearly collinear), $A^T A$ squares the condition number ($\kappa(A^T A) \approx \kappa(A)^2$), making the solution sensitive to floating-point errors.

## 3. Implementation Details

The implementation is located in `train.py` and relies on `my_math/`.

- **Custom Matrix Class:** Supports Matrix-Matrix multiplication, Transposition (`.T`), and Gaussian Elimination (`.solve()`).
- **Validation:** The solver raises `ValueError` if a singular matrix is detected (0 pivot).

### Limitations

- **Not optimized for sparse matrices:** Dense storage is used.
- **Numerical Precision:** Python's standard `float` (64-bit) is used; no arbitrary precision support.

## 4. Usage & Verification

To run the implementation on synthetic linear data:

```bash
python train.py
```

### Example: Fitting a Line

Given observations: `(1, 2), (2, 4), (3, 5)`

Design Matrix: A = [[1, 1], [1, 2], [1, 3]]
Target Vector: b = [2, 4, 5]

Solution: x = [0.6667, 1.5000] ≈ y = 1.5x + 0.67

## 5. Reference

**Legendre, A.-M** (1805). _Nouvelles méthodes pour la détermination des orbites des comètes._ Paris: Firmin Didot. [Available via Gallica](https://gallica.bnf.fr/ark:/12148/bpt6k15209372/f84.item)
