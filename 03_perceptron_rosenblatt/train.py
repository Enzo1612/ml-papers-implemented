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

        if isinstance(activation, (int, float)):
            return 1 if activation > 0 else -1
        elif isinstance(activation, Vector):   
            return activation.apply_func(lambda v: 1 if v > 0 else -1)
        else:
            raise TypeError("activation should be a float or a Vector")
    
    def train(self, X: Matrix, y: Vector):
        for _ in range(self.n_iterations):
            for i in range(len(X.rows)):
                current_x = X.rows[i]
                true_label = y[i]

                prediction = self.predict(current_x)
                
                if isinstance(prediction, Vector):
                    prediction = prediction[0]

                error = true_label - prediction

                if error != 0:
                    update_val = self.learning_rate * error
                    self.weights = self.weights + (current_x * update_val)

                    self.bias = self.bias + update_val
                