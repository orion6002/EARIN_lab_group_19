from ast import If

from anyio import current_effective_deadline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def function(x, y):
    return 2 * np.sin(x) + 3 * np.cos(y)
    # z = 2.sin(x) + 3.cos(y)

def newton_method(initial_guess, alpha, tol=1e-6, max_iter=1000):
    """
    Newton method

    Parameters:
    - initial_guess: initial 2D coordinate vector
    - alpha: step size parameter
    - tol: tolerance, convergence criteria
    - max_iter: maximum number of iterations
    """

    dx, dy = 0, 0
    Hf_inv = [[0, 0], [0, 0]]
    dk = [0, 0]

    current_guess = initial_guess
    for i in range (max_iter):
        dx = 2 * np.cos(current_guess[0])
        dy = -3 * np.sin(current_guess[1])
        if np.sqrt(dx**2 + dy**2) > tol :
            Hf_inv = np.linalg.inv([[-2*np.sin(dx), 0], [0, -3*np.cos(dy)]])
            dk = -1 * np.matmul(Hf_inv, [dx, dy])
            # we have to check if the current_guess is not out of bounds
            if abs(current_guess[0] + alpha * dk[0]) > max_abs_range[0] or abs(current_guess[1] + alpha * dk[1]) > max_abs_range[1] :
                current_guess[0], current_guess[1] = (current_guess[0] + alpha * dk[0]), (current_guess[1] + alpha * dk[1])
                return (current_guess, i+1)
            current_guess[0], current_guess[1] = current_guess[0] + alpha * dk[0], current_guess[1] + alpha * dk[1]
        else:
            return (current_guess, i+1)
    return (current_guess, max_iter)


def visualize():
    """
    Visualization function: creates 3D plot of the function. Use colors to show the Z-coordinate
    """


# visualize()

# Example usage:
max_abs_range = [5.5, 5.5]
initial_guess_1 = [2.0, 2.0]
print(f"test {initial_guess_1} test\n")
learning_rate_1 = 0.1
minimum_1, iterations_1 = newton_method(initial_guess_1, learning_rate_1)


print(
    f"Minimum approximation with initial guess {initial_guess_1}: {minimum_1}, Iterations: {iterations_1}"
)
