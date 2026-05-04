from abc import abstractmethod, ABC
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


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
        self.weight = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
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

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error loss."""
    return np.mean((y_pred - y_true) ** 2)


def mse_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Derivative of MSE with respect to y_pred."""
    return (2 / y_pred.size) * (y_pred - y_true)


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Absolute Error loss."""
    return np.mean(np.abs(y_pred - y_true))


def mae_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Derivative of MAE with respect to y_pred."""
    return np.sign(y_pred - y_true) / y_pred.size


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax applied row-wise."""
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(shifted_x)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Cross-entropy loss for one-hot encoded labels.

    y_pred contains raw logits. Softmax is applied inside the loss.
    """
    probs = softmax(y_pred)
    eps = 1e-12
    return -np.mean(np.sum(y_true * np.log(probs + eps), axis=1))


def cross_entropy_derivative(y_pred: np.ndarray,
                             y_true: np.ndarray) -> np.ndarray:
    """
    Derivative of softmax + cross-entropy with respect to logits.

    This combined derivative is more stable than differentiating softmax
    and cross-entropy separately.
    """
    return (softmax(y_pred) - y_true) / y_pred.shape[0]


mse_loss = Loss(mse, mse_derivative)
ce_loss = Loss(cross_entropy, cross_entropy_derivative)
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
            loss: Loss = None) -> List[float]:
        if loss is not None:
            self.loss = loss

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
                print(f"Epoch {epoch}: loss={float(loss_val):.4f}")
    
        return loss_history

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, output_error_derivative):
        return output_error_derivative * (self.input > 0)




def load_mnist_subset(train_size: int = 10000,
                      test_size: int = 2000,
                      seed: int = 0):
    """Load MNIST, normalize inputs and one-hot encode labels."""
    dataset = fetch_openml("mnist_784", version=1, as_frame=False)

    x = dataset.data.astype(np.float32) / 255.0
    y = dataset.target.astype(int)
    y_onehot = np.eye(10)[y]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_onehot,
        train_size=train_size,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    return x_train, x_test, y_train, y_test


def create_network(architecture: List[int], learning_rate: float) -> Network:
    """
    Create a ReLU MLP.

    The final layer is linear because Cross-Entropy applies softmax internally.
    """
    layers = []

    for i in range(len(architecture) - 2):
        layers.append(FullyConnected(architecture[i], architecture[i + 1]))
        layers.append(ReLU())

    layers.append(FullyConnected(architecture[-2], architecture[-1]))

    return Network(layers, learning_rate)


def accuracy(network: Network,
             x_data: np.ndarray,
             y_data: np.ndarray) -> float:
    """Compute classification accuracy."""
    y_pred = network(x_data)
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_data, axis=1)
    return float(np.mean(pred_labels == true_labels))


def run_experiments():
    """Run Group D experiments: MSE vs MAE vs Cross-Entropy."""
    seeds = [0, 1, 2]
    epochs = 30
    learning_rate = 0.1

    architectures = {
        "784-128-10": [784, 128, 10],
        "784-256-128-10": [784, 256, 128, 10],
    }

    losses = {
        "MSE": mse_loss,
        "MAE": mae_loss,
        "Cross-Entropy": ce_loss,
    }

    x_train, x_test, y_train, y_test = load_mnist_subset()
    results = {}
    histories = {}

    for arch_name, architecture in architectures.items():
        results[arch_name] = {}
        histories[arch_name] = {}

        for loss_name, loss_obj in losses.items():
            accuracies = []
            seed_histories = []

            for seed in seeds:
                np.random.seed(seed)
                network = create_network(architecture, learning_rate)
                network.compile(loss_obj)

                history = network.fit(
                    x_train,
                    y_train,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    verbose=0,
                )

                test_acc = accuracy(network, x_test, y_test)
                accuracies.append(test_acc)
                seed_histories.append(history)

                print(
                    f"Architecture={arch_name}, "
                    f"Loss={loss_name}, "
                    f"Seed={seed}, "
                    f"Test accuracy={test_acc:.4f}"
                )

            results[arch_name][loss_name] = {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies)),
            }
            histories[arch_name][loss_name] = np.mean(seed_histories, axis=0)

    print("\nMean ± std test accuracy")
    for arch_name, loss_results in results.items():
        print(f"\nArchitecture: {arch_name}")
        for loss_name, stats in loss_results.items():
            print(
                f"{loss_name}: "
                f"{stats['mean']:.4f} ± {stats['std']:.4f}"
            )

    for arch_name, loss_histories in histories.items():
        plt.figure(figsize=(8, 5))
        for loss_name, mean_history in loss_histories.items():
            plt.plot(mean_history, label=loss_name)

        plt.title(f"Training loss curves - Architecture {arch_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"loss_curves_{arch_name}.png", dpi=300)
        plt.show()

    return results, histories


if __name__ == "__main__":
    run_experiments()