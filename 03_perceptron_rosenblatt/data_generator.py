from typing import List
from my_math.vector import Vector
from my_math.matrix import Matrix
import random

def generate_data(n_samples: int) -> tuple[Matrix, List[int]]:

    w_true = Vector([random.uniform(-1, 1), random.uniform(-1, 1)])
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
    
    return Matrix(X), y