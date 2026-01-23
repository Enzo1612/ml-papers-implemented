from my_math.vector import Vector
from my_math.matrix import Matrix
class Perceptron:
    def __init__(self, input_dim: int, learning_rate: float = 0.1, n_iterations: int = 100, bias: float = 0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = Vector([0.0] * input_dim)
        self.bias = bias
    

    def predict(self, x):
        if isinstance(x, Matrix):
            z = x @ self.weights
        elif isinstance(x, Vector):
            z = x.dot(self.weights)
        else:
            raise TypeError("x should be a Vector or a Matrix.")
        
        activation = z + self.bias

        if isinstance(z, (int, float)):
            return 1 if activation > 0 else -1
        elif isinstance(activation, Vector):   
            return activation.apply_func(lambda v: 1 if v > 0 else -1)
    
