"""A module containing optimization algorithms.

It serves as a library for optimization-related tasks.
"""

import dataclasses
import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt


class GoldenSectionSearch:
    """A class that performs golden section search to find the minimum of a unimodal function.

    The search is modified such that the initial point can be either the 'a' or 'b' point.
    """
    G = (math.sqrt(5) - 1) / 2  # 1 / phi
    G2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2

    def __init__(self, func: Callable[[float], float], tol: float = 0.003):
        self.func = func
        self.tol = tol

    @dataclasses.dataclass
    class Variables:
        lower_bound: float
        upper_bound: float
        a: float
        b: float
        f_a: float
        f_b: float
        remaining_steps: int
        range: float

    def _verify_point_locations(self, var: Variables):
        assert var.lower_bound < var.a < var.b < var.upper_bound
        interval = var.upper_bound - var.lower_bound
        np.testing.assert_almost_equal(interval, var.range)
        np.testing.assert_almost_equal(var.lower_bound + self.G2 * interval, var.a)
        np.testing.assert_almost_equal(var.lower_bound + self.G * interval, var.b)
        np.testing.assert_almost_equal(var.a + self.G * interval, var.upper_bound)
        np.testing.assert_almost_equal(var.b + self.G2 * interval, var.upper_bound)

    def search_with_range(self, search_range: float, init_a: float = None, init_b: float = None) -> tuple[float, float]:
        """Performs golden section search with given range and initial a or b point.

        Returns:
            A tuple of (x, f(x)) where x is the estimated location of the minimum and f(x) is the function value at x.
        """
        if (init_a is None) == (init_b is None):
            raise ValueError("Exactly one of init_a and init_b must be provided.")

        # Check the range and translate it to number of steps.
        if search_range <= self.tol:
            xc = init_a if init_a is not None else init_b
            func_xc = self.func(xc)
            return xc, func_xc

        # Number of steps required to reach the tolerance
        n_steps = int(math.ceil(math.log(self.tol / search_range) / math.log(self.G)))

        if init_a is not None:
            lower_bound = init_a - self.G2 * search_range
            upper_bound = lower_bound + search_range
            init_b = lower_bound + self.G * search_range
        else:
            upper_bound = init_b + self.G2 * search_range
            lower_bound = upper_bound - search_range
            init_a = upper_bound - self.G * search_range
        var = self.Variables(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            a=init_a,
            b=init_b,
            f_a=self.func(init_a),
            f_b=self.func(init_b),
            remaining_steps=n_steps,
            range=search_range,
        )
        self._verify_point_locations(var)

        # Run the search iteration.
        while self._next(var):
            pass
        if var.f_a < var.f_b:
            return var.a, var.f_a
        else:
            return var.b, var.f_b

    def _next(self, var: Variables):
        if var.remaining_steps <= 0:
            return False
        var.remaining_steps -= 1

        if var.f_a < var.f_b:
            # var.lower_bound is unchanged
            var.upper_bound = var.b
            var.b = var.a
            var.f_b = var.f_a
            var.range *= self.G
            var.a = var.lower_bound + self.G2 * var.range
            var.f_a = self.func(var.a)
        else:
            var.lower_bound = var.a
            # var.upper_bound is unchanged
            var.a = var.b
            var.f_a = var.f_b
            var.range *= self.G
            var.b = var.lower_bound + self.G * var.range
            var.f_b = self.func(var.b)
        return True


def nelder_mead(
    func: Callable[[Sequence[float]], float],
    x0: Sequence[float] | None = None,
    initial_simplex: Sequence[Sequence[float]] | None = None,
    max_iter: int = 10,
    max_eval: int = 10,
) -> dict[str, Sequence[float] | float]:
    """Minimizes a scalar function using Nelder-Mead algorithm.

    Args:
        func:  A scalar function to be minimized.
        x0:  The initial point. If the initial simplex is not given, a new simplex will be created around x0.
        initial_simplex:  A sequence of vertices of the initial simplex. If vertice have dimension n_dim, then the
            length of the sequence should be (n_dim + 1).
        max_iter:  maximum number of iterations before the algorithm termination.
        max_eval:  maximum number of func evaluations before the algorithm termination. This number might not be kept
            all the time and can sometimes be surpassed by an additional (small) constant number of function evaluation.

    Returns:
        A dictionary containing the minimum point returned by the algorithm (result['x']) and the function evaluation
        at that point (result['func']).
    """
    if (x0 is None) and (initial_simplex is None):
        raise ValueError("Starting point or initial simplex are necessary")

    # Scipy params
    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5

    # Define starting point and initial simplex
    if x0 is None:
        x0 = np.array(initial_simplex[0], dtype=np.float64)
    else:
        x0 = np.array(x0, dtype=np.float64)
    n_dim = len(x0)

    if initial_simplex is None:
        initial_simplex = np.zeros((n_dim + 1, n_dim), np.float64)
        initial_simplex[1:, :] = 0.1 * np.eye(n_dim)
        initial_simplex += x0
    else:
        initial_simplex = np.array(initial_simplex, dtype=np.float64)
        if initial_simplex.shape != (n_dim + 1, n_dim):
            raise ValueError("The shape of the initial simplex is not correct.")

    # Stopping criteria parameters
    iters = 0
    n_evals = 0

    # Initialize container for function evaluation storage
    # Store simplex and evaluations in the simplex vertices
    sim_eval = np.zeros(n_dim + 1, dtype=float)
    simplex = np.zeros_like(initial_simplex)
    for i, point in enumerate(initial_simplex):
        simplex[i] = point
        sim_eval[i] = func(point)
        n_evals += 1

    # Sort based no function evaluation
    sorted_idx = sim_eval.argsort()
    simplex = simplex[sorted_idx]
    sim_eval = sim_eval[sorted_idx]

    # Iterate amoeba algorithm
    while iters < max_iter and n_evals < max_eval:
        iters += 1
        if_shrink = False

        # Calculate the center of the simplex' best edge
        sim_center = np.sum(simplex[:-1], axis=0) / n_dim

        # REFLECTION
        point_reflected = sim_center + rho * (sim_center - simplex[-1])
        reflected_eval = func(point_reflected)
        n_evals += 1

        if reflected_eval < sim_eval[0]:
            # If the new point is the best so far, expand

            # EXPANSION
            point_expanded = sim_center + rho * chi * (sim_center - simplex[-1])
            expanded_eval = func(point_expanded)
            n_evals += 1

            if expanded_eval < reflected_eval:
                simplex[-1] = point_expanded
                sim_eval[-1] = expanded_eval
            else:
                simplex[-1] = point_reflected
                sim_eval[-1] = reflected_eval
        else:
            # If the new reduced point is NOT the best so far, check some subcases

            if reflected_eval < sim_eval[-2]:
                simplex[-1] = point_reflected
                sim_eval[-1] = reflected_eval
            else:
                if reflected_eval < sim_eval[-1]:
                    # If the new point is not worse than all in the simplex

                    # CONTRACTION
                    point_contracted = sim_center + psi * rho * (sim_center - simplex[-1])
                    contracted_eval = func(point_contracted)
                    n_evals += 1

                    if contracted_eval <= reflected_eval:
                        simplex[-1] = point_contracted
                        sim_eval[-1] = contracted_eval
                    else:
                        if_shrink = True
                else:
                    # If the reflected point is worse than all in the simplex

                    # INSIDE CONTRACTION
                    point_inside = sim_center - psi * (sim_center - simplex[-1])
                    inside_eval = func(point_inside)
                    n_evals += 1

                    if inside_eval < sim_eval[-1]:
                        simplex[-1] = point_inside
                        sim_eval[-1] = inside_eval
                    else:
                        if_shrink = True

                if if_shrink:
                    # Shrink the whole simplex to gain precision
                    for j in range(1, n_dim + 1):
                        simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                        sim_eval[j] = func(simplex[j])
                        n_evals += 1

        # Sort based no function evaluation
        sorted_idx = sim_eval.argsort()
        simplex = simplex[sorted_idx]
        sim_eval = sim_eval[sorted_idx]

    # Return optimal point and the optimal function value
    result = {"x": simplex[0], "func": sim_eval[0]}

    return result
