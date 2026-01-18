from typing import List
from my_math.vector import Vector
from my_math.matrix import Matrix

dataset = [[1.0, 2.0], [2.0, 4.0], [3.0, 5.0]]

def get_design_matrix(dataset: List[List[float]]) -> Matrix:
    """
    Returns the design matrix (the A in Ax = b).

    Args:
        dataset (List[List[float]]): The set of observations.
    """
    return Matrix([[1, a[0]] for a in dataset])

def get_target_matrix(dataset: List[List[float]]) -> Vector:
    """
    Returns the target matrix (the b in Ax = b).

    Args:
        dataset (List[List[float]]): The set of observations.
    """
    return Vector([b[1] for b in dataset])

def solve_x(dataset: List[List[float]]):
    """
    Computes the solution to the equation Ax = b.

    Args:
        dataset (List[List[float]]): The set of observations.
    """
    design = get_design_matrix(dataset)
    target = get_target_matrix(dataset)
    return (design.transpose @ design).solve(design.transpose @ target)

result = solve_x(dataset=dataset)
print(f"{result[0]:.4f}, {result[1]:.4f}")