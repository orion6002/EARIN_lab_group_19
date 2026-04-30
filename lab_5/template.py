from abc import abstractmethod, ABC
from typing import List
import numpy as np


class Layer(ABC):
    """Basic building block of the Neural Network"""

    def __init__(self) -> None:
        self._learning_rate = 0.01

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through layer"""
        pass

    @abstractmethod
    def backward(self, output_error_derivative) -> np.ndarray:
        """Backward propagation of output_error_derivative through layer"""
        pass

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        assert learning_rate < 1, f"Given learning_rate={learning_rate} is larger than 1"
        assert learning_rate > 0, f"Given learning_rate={learning_rate} is smaller than 0"
        self._learning_rate = learning_rate


class FullyConnected(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.weight + self.bias

    def backward(self, output_error_derivative) -> np.ndarray:
        dw = self.input.T @ output_error_derivative
        db = np.sum(output_error_derivative, axis=0, keepdims=True)
        input_gradient = output_error_derivative @ self.weight.T

        self.weight -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        return input_gradient


# TODO (Group A / B): implement Tanh, Sigmoid, ReLU, LeakyReLU
# TODO (Group C):     implement Optimizer classes (SGD, Momentum, RMSProp, Adam)
#                     FullyConnected.backward should store gradients, not apply them
# TODO (Group D):     implement mse, mae, cross_entropy + softmax helpers
# TODO (Group E):     add lambda_ to FullyConnected for L2 decay; implement Dropout layer


class Loss:
    def __init__(self, loss_function: callable, loss_function_derivative: callable) -> None:
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.loss_function(y_pred, y_true)
    
    def loss_derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.loss_function_derivative(y_pred, y_true)

def mse(y_pred, y_true):
    return (2 / y_pred.size) * (y_pred - y_true)

def mse_derivative(y_pred, y_true):
    return (2 / y_pred.shape[0]) * (y_pred - y_true)

def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

def mae_derivative(y_pred, y_true):
    return np.sign(y_pred - y_true) / y_pred.shape[0]

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    probs = softmax(y_pred)
    return -np.mean(np.sum(y_true * np.log(probs + 1e-9), axis=1))

def cross_entropy_derivative(y_pred, y_true):
    return (softmax(y_pred) - y_true) / y_pred.shape[0]

mse_loss = Loss(mse, mse_derivative)
ce_loss  = Loss(cross_entropy, cross_entropy_derivative)
mae_loss = Loss(mae, mae_derivative)

class Network:
    def __init__(self, layers: List[Layer], learning_rate: float) -> None:
        self.layers = layers
        self.learning_rate = learning_rate

    def compile(self, loss: Loss) -> None:
        self.loss = loss
        for layer in self.layers:
            layer.learning_rate = self.learning_rate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through all layers"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            learning_rate: float,
            verbose: int = 0,
            loss = None) -> List[float]:
        if loss is not None:
            self.compile(loss)
        self.learning_rate = learning_rate
        self.compile(self.loss)
    
        loss_history = []
        for epoch in range(epochs):
            y_pred = self(x_train)
            loss_val = self.loss.loss(y_pred, y_train)
            loss_history.append(loss_val)
       
            grad = self.loss.loss_derivative(y_pred, y_train)
            for layer in reversed(self.layers):
                grad = layer.backward(grad)
        
            if verbose > 0 and epoch % verbose == 0:
                print(f"Epoch {epoch}: loss={loss_val:.4f}")
    
        return loss_history

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, output_error_derivative):
        return output_error_derivative * (self.input > 0)


# ---------------------------------------------------------------------------
# Data loading skeleton (shared by all groups)
# ---------------------------------------------------------------------------
# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split
#
# mnist = fetch_openml('mnist_784', version=1, as_frame=False)   # Group A / C / D / E
# # OR
# fmnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)  # Group B
#
# X = dataset.data / 255.0
# y = dataset.target.astype(int)
# y_onehot = np.eye(10)[y]
# X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=0)