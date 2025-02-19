"""
nano_bo: A minimal Bayesian Optimization implementation
Inspired by Andrej Karpathy's nanogpt

Key Features:
-------------
This implementation includes:
- Farthest Point Sampling for initial points
- Gaussian Process (GP) as surrogate model
- Expected Improvement (EI) as acquisition function
- Compound kernel (RBF + White Noise) with ARD
- L-BFGS-B with random restarts for acquisition optimization
- Minimal Active Learning mode (interactive labeling) if no function is provided

Mathematical Background:
-----------------------
Bayesian Optimization (BO) is a sequential design strategy for global optimization of black-box functions.
Key components:

1. Surrogate Model (Gaussian Process):
   - Models the objective function f(x) as a Gaussian Process
   - Provides mean μ(x) and variance σ²(x) estimates at any point
   - Uses kernel functions to define function smoothness/correlation

2. Acquisition Function (Expected Improvement):
   - Balances exploration (high uncertainty) vs exploitation (low mean)
   - EI(x) = E[max(0, f_best - f(x))]
   - Closed form: EI(x) = (f_best - μ(x))Φ(Z) + σ(x)φ(Z)
   where Z = (f_best - μ(x))/σ(x), Φ is CDF, φ is PDF of standard normal

3. Optimization Process:
   a. Initial Design: Sample points using space-filling design (FPS)
   b. Iterate until budget exhausted:
      - Fit GP to observed data
      - Optimize acquisition function
      - Evaluate new point
      - Update observations
   c. Return best observed point

Author: Tim-Lin - DeepBioLab
Date: 2025-02-19
License: MIT
"""

import numpy as np
from typing import Callable, Optional, Tuple, List, Union
from dataclasses import dataclass

from tqdm import tqdm
from scipy.stats import norm
from scipy.optimize import minimize


@dataclass
class Config:
    """Configuration for Bayesian Optimization"""
    # for basic BO settings
    n_iterations: int = 50  # number of BO iterations
    init_points: int = 15  # number of initial points for GP initialization
    init_method: str = "fps"  # method of sampling of initial points

    # for surrogate function
    noise_level: float = 1e-6  # noise level for GP
    length_scale: float = None  # RBF kernel length scale (per dimension)

    # for acquisition function
    xi: float = 0.01  # exploration-exploitation trade-off in EI
    acq_samples: int = 10000  # number of samples for acquisition optimization
    acq_batch_size: int = 1000  # batch size for acquisition function evaluation

    # for acquisition function optimization
    n_restarts: int = 10  # number of restart points for L-BFGS-B
    max_optim_iter: int = 100  # maximum iterations for each L-BFGS-B run

    # for reproducibility
    random_state: int = 42  


def farthest_point_sampling(
    n_points: int, bounds: "np.ndarray", n_candidates: int = 1000
) -> "np.ndarray":
    """
    Generate initial sample points using farthest point sampling method
    
    Algorithm:
    ----------
    1. Initialize:
       - Select first point uniformly at random within bounds
    
    2. Iterative Selection:
       For each remaining point:
       a. Generate n_candidates random points
       b. For each candidate:
          - Compute minimum distance to existing points
       c. Select candidate with maximum minimum distance
    
    Mathematical Background:
    -----------------------
    - Maximizes minimum distance between points
    - Approximates minimax facility location
    - Provides better space-filling than pure random sampling
    """
    dim = bounds.shape[0]
    samples = []

    # Generate first point: uniform sampling in each dimension
    first_sample = np.random.rand(dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    samples.append(first_sample)

    # Iteratively select remaining n_points-1 points
    for _ in range(1, n_points):
        # Randomly generate n_candidates points within bounds
        candidates = (
            np.random.rand(n_candidates, dim) * (bounds[:, 1] - bounds[:, 0])
            + bounds[:, 0]
        )
        # Calculate Euclidean distances between candidates and existing samples
        distances = []
        for candidate in candidates:
            # For current candidate, compute distances to all existing samples and take minimum
            dists = np.linalg.norm(np.array(samples) - candidate, axis=1)
            distances.append(np.min(dists))
        distances = np.array(distances)
        # Select candidate that is farthest from existing samples
        best_candidate = candidates[np.argmax(distances)]
        samples.append(best_candidate)

    return np.array(samples)


class Kernel:
    """Base kernel class"""

    def __init__(self):
        pass

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RBFKernel(Kernel):
    """
    Radial Basis Function (RBF) / Gaussian Kernel with Automatic Relevance Determination (ARD)
    
    Mathematical form:
    k(x, x') = exp(-0.5 * Σᵢ ((xᵢ - x'ᵢ)²/l²ᵢ))
    
    where:
    - x, x' are input vectors
    - lᵢ is the length scale for dimension i (controls smoothness)
    - ARD allows different length scales per dimension
    
    Properties:
    - Stationary: depends only on distance between points
    - Universal: can approximate any continuous function
    - Infinitely differentiable: produces smooth functions
    """

    def __init__(
        self,
        length_scale: Union[float, np.ndarray] = 0.1,
        length_scale_bounds: Tuple[float, float] = (1e-3, 1e3),
    ):
        super().__init__()
        self.length_scale = (
            length_scale
            if isinstance(length_scale, np.ndarray)
            else np.array([length_scale])
        )
        self.length_scale_bounds = length_scale_bounds

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF kernel with ARD if length_scale is array"""
        # Reshape length_scale for broadcasting
        ls = self.length_scale.reshape(1, -1)

        # Scale inputs by length_scale
        X1_scaled = X1 / ls
        X2_scaled = X2 / ls

        # Compute squared Euclidean distances
        X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
        cross_term = np.dot(X1_scaled, X2_scaled.T)
        squared_dist = X1_norm + X2_norm - 2 * cross_term

        return np.exp(-0.5 * squared_dist)


class WhiteKernel(Kernel):
    """White noise kernel with bounds"""

    def __init__(
        self,
        noise_level: float = 1e-2,
        noise_level_bounds: Tuple[float, float] = (1e-2, 1e1),
    ):
        super().__init__()
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute white noise kernel"""
        if X1 is X2:  # Using 'is' to check if they're the same object
            return self.noise_level * np.eye(X1.shape[0])
        return np.zeros((X1.shape[0], X2.shape[0]))


class CompoundKernel(Kernel):
    """Compound kernel that combines multiple kernels"""

    def __init__(self, kernels: List[Tuple[float, Kernel]]):
        super().__init__()
        self.kernels = kernels  # List of (scale, kernel) tuples

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute compound kernel"""
        return sum(scale * kernel(X1, X2) for scale, kernel in self.kernels)


class GaussianProcess:
    """
    Gaussian Process Regression (GPR) Implementation
    
    Mathematical Background:
    -----------------------
    1. Prior: f(x) ~ GP(0, k(x,x'))
    
    2. Posterior Distribution:
       μ(x*) = K*ᵀK⁻¹y
       σ²(x*) = K** - K*ᵀK⁻¹K*
       
       where:
       - K: kernel matrix for training points [n x n]
       - K*: kernel between test and training points [n x 1]
       - K**: kernel between test point and itself [1 x 1]
       - y: training observations
       
    3. Implementation Notes:
       - Uses Cholesky decomposition for numerical stability
       - Adds jitter to diagonal for conditioning
       - Handles multi-dimensional inputs via ARD kernels
    """
    def __init__(self, kernel: Optional[Kernel] = None):
        if kernel is None:
            # Default kernel: RBF + White noise
            self.kernel = CompoundKernel(
                [
                    (
                        1.0,
                        RBFKernel(length_scale=1e-1, length_scale_bounds=(1e-3, 1e3)),
                    ),
                    (
                        1.0,
                        WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-2, 1e1)),
                    ),
                ]
            )
        else:
            self.kernel = kernel

        self.X_train = None
        self.y_train = None
        self.K = None
        self.K_inv = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP to training data"""
        self.X_train = X
        self.y_train = y

        # Compute kernel matrix
        self.K = self.kernel(X, X)

        # Add small jitter for numerical stability
        jitter = 1e-10
        self.K += jitter * np.eye(len(X))

        try:
            self.K_inv = np.linalg.inv(self.K)
        except np.linalg.LinAlgError:
            # If inversion fails, try with larger jitter
            jitter = 1e-6
            self.K += jitter * np.eye(len(X))
            self.K_inv = np.linalg.inv(self.K)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at test points"""
        K_star = self.kernel(X_test, self.X_train)
        K_star_star = self.kernel(X_test, X_test)

        # Compute posterior mean
        mean = K_star.dot(self.K_inv).dot(self.y_train)

        # Compute posterior variance
        var = K_star_star - K_star.dot(self.K_inv).dot(K_star.T)
        var = np.diag(var).reshape(-1, 1)

        # Ensure non-negative variance
        var = np.maximum(var, 1e-10)

        return mean, var


class ExpectedImprovement:
    """
    Expected Improvement (EI) Acquisition Function
    
    Mathematical Formulation:
    ------------------------
    EI(x) = E[max(0, f_best - f(x))]
    
    Closed form solution:
    EI(x) = (f_best - μ(x))Φ(Z) + σ(x)φ(Z)
    
    where:
    - Z = (f_best - μ(x) - ξ)/σ(x)
    - Φ: standard normal CDF
    - φ: standard normal PDF
    - ξ: exploration-exploitation trade-off parameter
    - f_best: current best observed value
    - μ(x): GP posterior mean at x
    - σ(x): GP posterior standard deviation at x
    """

    def __init__(self, xi: float = 0.01):
        self.xi = xi

    def __call__(
        self, mean: np.ndarray, sigma: np.ndarray, y_best: float
    ) -> np.ndarray:
        """Compute Expected Improvement"""
        # Standardize inputs
        with np.errstate(divide="warn"):
            imp = mean - y_best - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei


class BayesianOptimization:
    def __init__(
        self, bounds: np.ndarray, f: Optional[Callable] = None, config: Optional[Config] = None
    ):
        """
        If f is provided, standard function optimization mode is used.
        If f is None, active learning mode is enabled (labels are provided interactively).
        """
        self.f = f
        self.bounds = bounds
        self.config = config or Config()

        # Initialize kernel
        length_scale = (
            self.config.length_scale
            if self.config.length_scale is not None
            else np.array([0.1 * (b[1] - b[0]) for b in bounds])
        )
        noise_level = (
            self.config.noise_level if self.config.noise_level is not None else 1e-2
        )

        # Initialize kernels with proper length scales
        if isinstance(length_scale, np.ndarray):
            # Multi-dimensional case: one length scale per dimension
            rbf_kernel = RBFKernel(
                length_scale=length_scale, length_scale_bounds=(1e-3, 1e3)
            )
        else:
            # Single dimension case: scalar length scale
            rbf_kernel = RBFKernel(
                length_scale=float(length_scale), length_scale_bounds=(1e-3, 1e3)
            )

        white_kernel = WhiteKernel(
            noise_level=noise_level, noise_level_bounds=(1e-2, 1e1)
        )

        kernel = CompoundKernel([(1.0, rbf_kernel), (1.0, white_kernel)])

        # Initialize GP with compound kernel
        self.gp = GaussianProcess(kernel=kernel)
        self.ei = ExpectedImprovement(xi=self.config.xi)

        # Set random state for reproducibility
        np.random.seed(self.config.random_state)

        # Initialize history
        self.X_samples = []
        self.y_samples = []
        self.best_value = float("inf")
        self.best_params = None

    def query(self, x: np.ndarray) -> float:
        """If no function is provided, ask the user for a label."""
        label = input(f"\n>>> Provide label for sample {x}: ")
        return float(label)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate sample x using the provided function or via interactive query."""
        return self.f(x) if self.f is not None else self.query(x)

    def _sample_initial_points(self, n_points: int = 5, method: str = "fps") -> None:
        """Generate initial sampling points.

        Args:
            n_points: Number of initial sample points to generate
            method: Sampling method, "fps" for farthest point sampling, "random" for random sampling (np.random.rand)
        """
        dim = self.bounds.shape[0]
        if method == "fps":
            # Generate initial points using farthest point sampling
            samples = farthest_point_sampling(n_points, self.bounds, n_candidates=1000)
        elif method == "random":
            # Generate initial points using random sampling and scale to bounds
            samples = np.random.rand(n_points, dim)
            for i in range(dim):
                samples[:, i] = (
                    samples[:, i] * (self.bounds[i, 1] - self.bounds[i, 0])
                    + self.bounds[i, 0]
                )
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # Evaluate generated initial points
        for x in samples:
            y = self.evaluate(x)
            self.X_samples.append(x)
            self.y_samples.append(y)

            if y < self.best_value:
                self.best_value = y
                self.best_params = x

    def optimize(self) -> Tuple[np.ndarray, float]:
        """Main optimization loop"""
        # Sample initial points
        self._sample_initial_points(
            n_points=self.config.init_points, method=self.config.init_method
        )

        X = np.array(self.X_samples)
        y = np.array(self.y_samples).reshape(-1, 1)

        for _ in tqdm(
            range(self.config.n_iterations), desc="Optimizing", ncols=80, leave=True
        ):
            # Fit GP to current data
            self.gp.fit(X, y)

            # Optimize acquisition function
            x_next = self._optimize_acquisition()

            # Evaluate new point (either via function call or interactive query)
            y_next = self.evaluate(x_next)

            # Update data
            X = np.vstack((X, x_next))
            y = np.vstack((y, y_next))

            # Update best observation
            if y_next < self.best_value:
                self.best_value = y_next
                self.best_params = x_next

            self.X_samples.append(x_next)
            self.y_samples.append(y_next)

        return self.best_params, self.best_value

    def _optimize_acquisition(self) -> np.ndarray:
        """
        Optimize acquisition function using multi-start optimization strategy
        
        Algorithm:
        ----------
        1. Initial Search Phase:
            a. Generate n_random candidate points uniformly
            b. Process in batches for memory efficiency
            c. Evaluate EI for all candidates
            d. Select top n_starts points as potential starts
        
        2. Local Optimization Phase:
            a. Run L-BFGS-B from each starting point
            b. Use gradient-based optimization since EI is differentiable
            c. Handle bounds via L-BFGS-B's built-in constraint handling
        
        3. Selection:
            Return best point found across all local optimizations
        
        Implementation Notes:
        --------------------
        - Uses batching to handle large number of initial candidates
        - Multiple restarts help avoid poor local optima
        - Bounds are strictly enforced throughout optimization
        """
        dim = self.bounds.shape[0]
        n_random = self.config.acq_samples
        batch_size = self.config.acq_batch_size
        n_starts = self.config.n_restarts  # Number of start points for L-BFGS-B

        # Generate random points
        X_random = np.random.rand(n_random, dim)
        for i in range(dim):
            X_random[:, i] = (
                X_random[:, i] * (self.bounds[i, 1] - self.bounds[i, 0])
                + self.bounds[i, 0]
            )

        # Process in batches for initial search
        best_points = []
        best_ei_values = []

        for i in range(0, n_random, batch_size):
            batch = X_random[i : i + batch_size]
            mean, var = self.gp.predict(batch)
            ei_values = self.ei(mean, np.sqrt(var), self.best_value)

            # Get top n_starts points from this batch
            batch_top_indices = np.argsort(ei_values.ravel())[-n_starts:]
            best_points.extend(batch[batch_top_indices])
            best_ei_values.extend(ei_values.ravel()[batch_top_indices])

        # Convert to numpy arrays for easier handling
        best_points = np.array(best_points)
        best_ei_values = np.array(best_ei_values)

        # Select top n_starts points overall
        top_indices = np.argsort(best_ei_values)[-n_starts:]
        start_points = best_points[top_indices]

        # Define objective for L-BFGS-B (minimize negative EI)
        def objective(x):
            x = x.reshape(1, -1)
            mean, var = self.gp.predict(x)
            ei = self.ei(mean, np.sqrt(var), self.best_value)
            return -ei[0]

        # Run L-BFGS-B from multiple starting points
        results = []
        for sp in start_points:
            res = minimize(
                objective,
                sp,
                method="L-BFGS-B",
                bounds=self.bounds,
                options={"ftol": 1e-6, "maxiter": self.config.max_optim_iter},
            )
            results.append((res.fun, res.x))
        best_result = min(results, key=lambda x: x[0])
        return best_result[1]


if __name__ == "__main__":
    from plots import plot_1d_results, plot_2d_results

    # =======================================================================
    # ================ 1D Optimization with Explicit Function ===============
    # =======================================================================
    print("Running 1D optimization with explicit function...")
    def objective_1d(x):
        """Simple 1D test function with multiple local minima."""
        return np.sin(3 * x) * x**2 + 0.7 * np.cos(2 * x)

    # Define bounds
    bounds_1d = np.array([[-2, 2]])

    # Create and run optimizer
    optimizer_1d = BayesianOptimization(
        bounds=bounds_1d,
        f=objective_1d,
        config=Config(
            n_iterations=20,
            length_scale=0.3,
            noise_level=1e-6,
            xi=0.01,
            init_points=5,
            init_method="random",
        ),
    )

    best_params_1d, best_value_1d = optimizer_1d.optimize()
    print(f"Best parameters: {best_params_1d}")
    print(f"Best value: {best_value_1d}")

    # Plot results
    plot_1d_results(optimizer_1d, bounds_1d, objective_1d)

    # =======================================================================
    # ================ 2D Optimization with Explicit Function ===============
    # =======================================================================
    print("\nRunning 2D optimization with explicit function...")
    def objective_2d(x):
        """
        Six-hump camel function.
        Global minimum: f(0.0898, -0.7126) = -1.0316
        """
        x1, x2 = x[0], x[1]
        term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2
        return term1 + term2 + term3

    # Define bounds
    bounds_2d = np.array([[-2, 2], [-2, 2]])

    # Create and run optimizer
    optimizer_2d = BayesianOptimization(
        bounds=bounds_2d,
        f=objective_2d,
        config=Config(
            n_iterations=50,
            length_scale=np.array([0.4, 0.4]),  # per-dimension length scales
            init_points=10,
            init_method="fps",
        ),
    )

    best_params_2d, best_value_2d = optimizer_2d.optimize()
    print(f"Best parameters found: {best_params_2d}")
    print(f"Best value found: {best_value_2d}")

    # Plot results
    plot_2d_results(optimizer_2d, bounds_2d, objective_2d)

    # =======================================================================
    # ================= 1D Optimization with Blackbox Problem ===============
    # =======================================================================
    print("\nRunning 1D optimization in active learning mode...")
    bounds_1d = np.array([[-2, 2]])
    # Create and run optimizer in active learning mode
    optimizer_1d_active = BayesianOptimization(
        bounds=bounds_1d,
        f=None,
        config=Config(
            n_iterations=5,
            init_points=3,
            init_method="random",
        ),
    )

    best_params_1d_active, best_value_1d_active = optimizer_1d_active.optimize()
    print(f"Best parameters (active): {best_params_1d_active}")
    print(f"Best value (active): {best_value_1d_active}")