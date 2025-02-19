import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(y_samples, best_value):
    """Plot convergence history"""
    best_values = np.minimum.accumulate(y_samples)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(y_samples, "b.", alpha=0.3, label="Observations")
    plt.plot(best_values, "r-", label="Best value")
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.legend()
    plt.title("Optimization History")

    plt.subplot(1, 2, 2)
    plt.hist(y_samples, bins=30, alpha=0.5)
    plt.axvline(best_value, color="r", linestyle="--", label="Best value")
    plt.xlabel("Objective value")
    plt.ylabel("Count")
    plt.title("Distribution of Observations")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_1d_results(optimizer, bounds, objective=None):
    """Plot results for 1D optimization."""
    x = np.linspace(bounds[0, 0], bounds[0, 1], 100).reshape(-1, 1)
    y = np.array([objective(xi) for xi in x]) if objective else None
    mean, var = optimizer.gp.predict(x)
    std = np.sqrt(var)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "k--", label="True function")
    plt.plot(x, mean, "b-", label="GP mean")
    plt.fill_between(
        x.ravel(),
        mean.ravel() - 2 * std.ravel(),
        mean.ravel() + 2 * std.ravel(),
        alpha=0.2,
    )
    plt.scatter(optimizer.X_samples, optimizer.y_samples, c="r", label="Observations")
    plt.scatter(
        [optimizer.best_params],
        [optimizer.best_value],
        c="g",
        marker="*",
        s=200,
        label="Best found",
    )
    plt.legend()
    plt.show()


def plot_2d_results(optimizer, bounds, objective=None, resolution=50):
    """Plot results for 2D optimization."""
    x1 = np.linspace(bounds[0, 0], bounds[0, 1], resolution)
    x2 = np.linspace(bounds[1, 0], bounds[1, 1], resolution)
    X1, X2 = np.meshgrid(x1, x2)

    # Compute true function if provided
    Y = np.zeros((resolution, resolution))
    if objective:
        for i in range(resolution):
            for j in range(resolution):
                Y[i, j] = objective(np.array([X1[i, j], X2[i, j]]))

    # GP prediction
    X_test = np.column_stack((X1.ravel(), X2.ravel()))
    mean, var = optimizer.gp.predict(X_test)
    mean = mean.reshape(resolution, resolution)

    Y = Y if objective else None

    """Plot 2D optimization results"""
    fig = plt.figure(figsize=(18, 6))

    # True function
    ax1 = fig.add_subplot(131, projection="3d")
    surf1 = ax1.plot_surface(X1, X2, Y, cmap="viridis", alpha=0.8)
    ax1.scatter(
        np.array(optimizer.X_samples)[:, 0],
        np.array(optimizer.X_samples)[:, 1],
        optimizer.y_samples,
        c="r",
        marker="o",
        s=100,
        label="Observations",
    )
    ax1.scatter(
        [optimizer.best_params[0]],
        [optimizer.best_params[1]],
        [optimizer.best_value],
        c="g",
        marker="*",
        s=200,
        label="Best found",
    )
    ax1.set_title("True Function")
    fig.colorbar(surf1, ax=ax1)

    # GP mean
    ax2 = fig.add_subplot(132, projection="3d")
    surf2 = ax2.plot_surface(X1, X2, mean, cmap="viridis", alpha=0.8)
    ax2.scatter(
        np.array(optimizer.X_samples)[:, 0],
        np.array(optimizer.X_samples)[:, 1],
        optimizer.y_samples,
        c="r",
        marker="o",
        s=100,
        label="Observations",
    )
    ax2.scatter(
        [optimizer.best_params[0]],
        [optimizer.best_params[1]],
        [optimizer.best_value],
        c="g",
        marker="*",
        s=200,
        label="Best found",
    )
    ax2.set_title("GP Mean")
    fig.colorbar(surf2, ax=ax2)

    # 2D contour with observations
    ax3 = fig.add_subplot(133)
    contour = ax3.contour(X1, X2, Y, levels=20)
    ax3.scatter(
        np.array(optimizer.X_samples)[:, 0],
        np.array(optimizer.X_samples)[:, 1],
        c="r",
        marker="o",
        s=100,
        label="Observations",
    )
    ax3.scatter(
        [optimizer.best_params[0]],
        [optimizer.best_params[1]],
        c="g",
        marker="*",
        s=200,
        label="Best found",
    )
    ax3.set_title("Contour Plot")
    fig.colorbar(contour, ax=ax3)

    plt.tight_layout()
    plt.show()
