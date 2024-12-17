import numpy as np
from lib.Player import Player


class SimpleNetwork:
    def __init__(self):
        # Initialize weights and biases
        self.W1 = np.random.randn(16, 6) * 0.01  # Hidden layer 1 weights (16 neurons, 6 inputs)
        self.b1 = np.zeros((16, 1))              # Hidden layer 1 biases
        self.W2 = np.random.randn(16, 16) * 0.01 # Hidden layer 2 weights (16 neurons, 16 inputs)
        self.b2 = np.zeros((16, 1))              # Hidden layer 2 biases
        self.W3 = np.random.randn(2, 16) * 0.01  # Output layer weights (2 outputs, 16 inputs)
        self.b3 = np.zeros((2, 1))               # Output layer biases
    
    def save(self, filename):
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    def load(self, filename):
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']

    def relu(self, x):
        return np.maximum(0, x)  # ReLU activation

    def tanh(self, x):
        return np.tanh(x)  # Tanh activation

    def forward(self, x):
        # Forward pass
        x = np.array(x).reshape(-1, 1)           # Reshape input to column vector
        z1 = self.W1 @ x + self.b1              # Hidden layer 1 linear transformation
        a1 = self.relu(z1)                      # Hidden layer 1 activation
        z2 = self.W2 @ a1 + self.b2             # Hidden layer 2 linear transformation
        a2 = self.relu(z2)                      # Hidden layer 2 activation
        z3 = self.W3 @ a2 + self.b3             # Output layer linear transformation
        output = self.tanh(z3)                  # Output layer activation
        return output.flatten()                 # Flatten to return throttle and steering

    def update_weights(self, grads, lr=0.001):

        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']
        self.W3 -= lr * grads['dW3']
        self.b3 -= lr * grads['db3']

    def backward(self, x, y_true):
        """
        Backpropagate gradients through the network.
        x: Input array (shape [6, 1]).
        y_true: Target output array (shape [2, 1]).
        """
        x = np.array(x).reshape(-1, 1)  # Ensure input is a column vector
        y_true = np.array(y_true).reshape(-1, 1)  # Ensure target is a column vector

        # Forward pass (to cache intermediate values)
        z1 = self.W1 @ x + self.b1
        a1 = self.relu(z1)
        z2 = self.W2 @ a1 + self.b2
        a2 = self.relu(z2)
        z3 = self.W3 @ a2 + self.b3
        y_pred = self.tanh(z3)

        # Compute loss gradient w.r.t output (MSE loss derivative)
        dL_dy = 2 * (y_pred - y_true)

        # Backpropagate through output layer
        dL_dz3 = dL_dy * (1 - y_pred**2)  # Tanh derivative
        dL_dW3 = dL_dz3 @ a2.T
        dL_db3 = dL_dz3

        # Backpropagate through second hidden layer
        dL_da2 = self.W3.T @ dL_dz3
        dL_dz2 = dL_da2 * (a2 > 0)  # ReLU derivative
        dL_dW2 = dL_dz2 @ a1.T
        dL_db2 = dL_dz2

        # Backpropagate through first hidden layer
        dL_da1 = self.W2.T @ dL_dz2
        dL_dz1 = dL_da1 * (a1 > 0)  # ReLU derivative
        dL_dW1 = dL_dz1 @ x.T
        dL_db1 = dL_dz1

        # Store gradients in a dictionary
        grads = {
            'dW1': dL_dW1,
            'db1': dL_db1,
            'dW2': dL_dW2,
            'db2': dL_db2,
            'dW3': dL_dW3,
            'db3': dL_db3,
        }

        return grads

def func(x):
    t = -1

    if (x[5] < 165.0/400.0 or x[2] > 250.0/400.0):
        t = 1
    c = x[0] + x[1] - x[3] - x[4]
    s = -c * 2.0

    return [t, s]


class SimpleAI(Player):

    def __init__(self, map, color, network=None):
        super().__init__(map, color)

        if (network):
            self.network = network
        else:
            self.network = SimpleNetwork()
            self.network.load("./players/model.npz")

        self.performance = 0

    def cast_rays(self):
        return [ self.car.cast_ray(ang, 5, 400) for ang in [ -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2 ] ]
    

    def control(self, rays):
        
        inputs = np.array([
            rays[0].distance / 400.0, 
            rays[1].distance / 400.0,  
            rays[2].distance / 400.0,  
            rays[3].distance / 400.0,  
            rays[4].distance / 400.0, 
            self.car.forward_velocity / self.car.max_forward_velocity])

        prediction = self.network.forward(inputs)
        # prediction = func(inputs)
        self.car.throttle = prediction[0]
        self.car.steer = prediction[1]

        self.performance += self.car.forward_velocity
