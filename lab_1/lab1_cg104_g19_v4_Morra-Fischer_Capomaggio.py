import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def function(x, y):
    return 2 * np.sin(x) + 3 * np.cos(y)

def newton_method(initial_guess, alpha, tol=1e-6, max_iter=1000):
    """
    Newton method

    Parameters:
    - initial_guess: initial 2D coordinate vector
    - alpha: step size parameter
    - tol: tolerance, convergence criteria
    - max_iter: maximum number of iterations
    """

    dx, dy = 0
    Hf = [[0, 0], [0, 0]]
    dk = [0, 0]
    ak = 0

    current_guess = initial_guess
    for i in range (max_iter):
        dx, dy = np.gradient(function(current_guess[0], current_guess[1]), current_guess[0], current_guess[1])
        if np.sqrt(dx**2 + dy**2) > tol :
            #generate Hessian Matrix [[][]] based on the current guess
            dk = -1 * np.matmul(Hf.np.linalg.inv(), [dx, dy])
            current_guess[0], current_guess[1] = current_guess[0] + ak * dk[0], current_guess[1] + ak * dk[1]
        else:
            return current_guess
    return current_guess


def visualize():
    """
    Visualization function: creates 3D plot of the function. Use colors to show the Z-coordinate
    """


visualize()

# Example usage:
initial_guess_1 = [2.0, 2.0]
learning_rate_1 = 0.1
minimum_1, iterations_1 = newton_method(initial_guess_1, learning_rate_1)


print(
    f"Minimum approximation with initial guess {initial_guess_1}: {minimum_1}, Iterations: {iterations_1}"
)
