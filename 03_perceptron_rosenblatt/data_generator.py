from typing import List, Optional
from my_math.vector import Vector
from my_math.matrix import Matrix
import random

def generate_data(n_samples: int, w_true: Optional[Vector] = None, b_true: Optional[float] = None) -> tuple[Matrix, List[int], Vector, float]:
    """Generate linearly separable data.
    
    Args:
        n_samples: Number of samples to generate
        w_true: True weight vector (if None, generates random)
        b_true: True bias (if None, generates random)
    
    Returns:
        X: Matrix of features
        y: List of labels (-1 or 1)
        w_true: The weight vector used
        b_true: The bias used
    """
    if w_true is None:
        w_true = Vector([random.uniform(-1, 1), random.uniform(-1, 1)])
    if b_true is None:
        b_true = random.uniform(-1, 1)

    X = []
    y = []
    while len(X) < n_samples:
        candidate = [random.uniform(-1, 1), random.uniform(-1, 1)]

        dot_prod = candidate[0] * w_true[0] + candidate[1] * w_true[1]
        val = dot_prod + b_true

        if abs(val) < 0.1:
            continue

        X.append(candidate)
        y.append(1 if val > 0 else -1)
    
    return Matrix(X), y, w_true, b_true