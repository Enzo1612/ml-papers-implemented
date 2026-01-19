from typing import List
from my_math.matrix import Matrix
from my_math.vector import Vector

def gradient_descent(
    iterations: int,
    dataset: List[List[float]],
    theta: Vector,
    A: Matrix,
    b: Vector,
    lr: float,
    tolerance: float = 1e-6
) -> tuple[Vector, List[float]]:
    """
    Implements Cauchy's Method of Steepest Descent (1847).
    
    Iteratively minimizes J(θ) = (1/2m)||Aθ - b||^2 by moving
    opposite to the gradient direction.
    
    Args:
        iterations: Maximum number of gradient steps
        dataset: Original (x, y) pairs for context
        theta: Initial parameter vector
        A: Design matrix (m x n)
        b: Target vector (m x 1)
        lr: Learning rate (step size)
        tolerance: Threshold for cost improvement per iteration.
            Stops when |J(t) - J(t-1)| < tolerance.
            Default 1e-6 ensures convergence to optimal solution.
    
    Returns:
        theta: Final optimized parameters
    """
    cost_history = []
    
    for i in range(iterations):
        prediction = A @ theta
        error = prediction - b
        cost = (error.dot(error)) / (2 * len(dataset))
        cost_history.append(cost)
        
        if i > 0 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"Converged at iteration {i}")
            break
        
        gradient = (A.transpose @ error) / len(dataset)
        theta = theta - lr * gradient
    
    return theta, cost_history

raw_data = [
    [1.0, 2],
    [2, 4],
    [3, 5]
]

# A = [[1, x_1]. [1, x_2], [1, x_3]] x_i = i^th feature

A_data = [
    [1.0, 1],
    [1, 2],
    [1, 3]
]

A = Matrix(A_data)

b_data = [2.0, 4.0, 5.0]
b = Vector(b_data)

initial_theta = Vector([0.0, 0.0])

learning_rate = 0.01
iterations = 10000

final_theta, cost_history = gradient_descent(
    dataset=raw_data,
    iterations=iterations, 
    A=A,
    b=b, 
    theta=initial_theta, 
    lr=learning_rate
)

print("Final Weights:", final_theta)
# Final Weights: [0.6666666325182643, 1.5000000150219412]