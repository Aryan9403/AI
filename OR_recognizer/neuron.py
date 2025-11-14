import numpy as np 

def sigmoid(z: float | np.ndarray) -> float | np.ndarray:

    """ sigmoid activation function
    maps any real number to (0, 1)"""
    return 1/(1 + np.exp(-z))

class Neuron:
    def __init__(self, n_inputs: int, *, weight_scale: float = 0.01):
        """"
        A single neuron with:
        - n_inputs: number of input features
        - weight_scale: scale for random weight initialization
        - scalar bias b

        """

        self.n_inputs = n_inputs
        #small random weights
        self.w = np.random.randn(n_inputs)*weight_scale
        #bias initialized to zero
        self.b = 0.0

    def forward(self, x: np.ndarray) -> float:
        """ computes activation for input x
        - x: shape (n_inputs)
        returns: scalar activation(single float calculated by single neuron where input can be a float or array of floats)
        z = np.dot(self.w, x) + self.b"""

        #safety check for input shape

        x = np.asarray(x, dtype=float)
        assert x.shape == (self.n_inputs,), f"Expected input shape ({self.n_inputs},), got {x.shape}"

        z = np.dot(self.w, x) + self.b
        a = sigmoid(z)

        return float(a)
    
    def train_on_example(self, x, y, lr=0.1):
        """ Trains a single example (x, y) with squared error loss
        x: shape(n_inputs)
        y: scalar 0 or 1"""
        x = np.asarray(x, dtype=float)
        y = float(y)
        assert x.shape == (self.n_inputs,), f"Expected input shape ({self.n_inputs},), got {x.shape}"

        a = self.forward(x)
        #compute gradients
        error = a - y
        loss = 0.5 * (error**2)

        # weight update
        self.w -= lr * error * x
        self.b -= lr * error

        return loss
    
    def __repr__(self) -> str:
        return f"Neuron(n_inputs={self.n_inputs}, weights={self.w}, bias={self.b})"