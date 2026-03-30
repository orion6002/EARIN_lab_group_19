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

    current_guess = np.array(initial_guess, dtype=float)
    # Store the successive points visited by Newton's method for visualization.
    path = [current_guess.copy()]
    
    for i in range(max_iter):
        x = current_guess[0]
        y = current_guess[1]

        # Compute the gradient of f at the current point.
        gradient = np.array([
            2 * np.cos(x),
            -3 * np.sin(y)
        ])

        grad_norm = np.linalg.norm(gradient)
        if grad_norm < tol:
            return current_guess.tolist(), i, path
        
        # Build the Hessian matrix of f at the current point.
        hessian = np.array([
            [-2 * np.sin(x), 0.0],
            [0.0, -3 * np.cos(y)]
        ])

        det_hessian = np.linalg.det(hessian)
        # Stop if the Hessian is too close to singular, because Newton's step
        # cannot be computed reliably in that case.
        if abs(det_hessian) < 1e-10:
            return current_guess.tolist(), i, path
        
        # Compute the Newton direction and update the current point.
        newton_direction = -np.linalg.inv(hessian) @ gradient
        current_guess = current_guess + alpha * newton_direction
        path.append(current_guess.copy())

    return current_guess.tolist(), max_iter, path

def visualize(examples):
    """
    Visualization function: creates 3D plot of the function. Use colors to show the Z-coordinate
    """
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the objective function surface over the required domain.
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)

    # Plot the trajectory obtained for each pair (initial_guess, alpha).
    for example in examples:
        initial_guess = example["initial_guess"]
        alpha = example["alpha"]

        # Run Newton's method and recover the visited points.
        minimum, iterations, path = newton_method(initial_guess, alpha)

        path = np.array(path)
        z_path = function(path[:, 0], path[:, 1])

        ax.plot(
            path[:, 0],
            path[:, 1],
            z_path,
            marker='o',
            markersize=4,
            linewidth=2,
            label=f"start={initial_guess}, alpha={alpha}"
        )

        ax.scatter(
            minimum[0],
            minimum[1],
            function(minimum[0], minimum[1]),
            s=60
        )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title("Newton's method trajectories")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.legend()
    plt.tight_layout()
    plt.show()

examples = [
    {"initial_guess": [4.0, 4.0], "alpha": 0.1},
    {"initial_guess": [2.0, 2.0], "alpha": 0.3},
    {"initial_guess": [-4.0, 2.0], "alpha": 0.5},
    {"initial_guess": [1.0, -4.0], "alpha": 0.8},
    {"initial_guess": [-2.5, -2.0], "alpha": 1.0},
]

visualize(examples)

for example in examples:
    minimum, iterations, path = newton_method(example["initial_guess"], example["alpha"])
    print(
        f"Minimum approximation with initial guess {example['initial_guess']} and alpha = {example['alpha']}: "
        f"{minimum}, Iterations: {iterations}"
    )