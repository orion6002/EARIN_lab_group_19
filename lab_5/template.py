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

        self.W -= self.learning_rate * dw
        self.b -= self.learning_rate * db
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
            verbose: int = 0) -> List[float]:
        pass


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