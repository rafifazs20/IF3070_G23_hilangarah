from time import perf_counter
from typing import Any

from .hill_climbing_helper import (
    HillClimbingResult,
    get_random_valid_neighbor,
    is_strictly_better,
)


def stochastic_hill_climbing(
    initial_state: Any,
    max_iterations: int = 1000,
    max_neighbor_attempts: int = 100,
) -> HillClimbingResult:
    """
    Run stochastic hill-climbing by sampling one valid neighbor per iteration.

    A sampled neighbor is accepted only when its objective value is strictly
    higher than the current state's objective value. The search stops when
    max_iterations is reached or valid-neighbor sampling fails.
    """
    if max_iterations < 0:
        raise ValueError("max_iterations tidak boleh negatif.")

    if max_neighbor_attempts < 1:
        raise ValueError("max_neighbor_attempts harus minimal 1.")

    start_time = perf_counter()

    initial = initial_state.copy()

    if not initial.is_valid():
        raise ValueError("Initial state harus memenuhi seluruh constraint.")

    current = initial.copy()
    objective_history = [float(current.evaluate())]

    iterations = 0
    accepted_moves = 0
    rejected_moves = 0
    sampling_failures = 0
    termination_reason = "max_iterations"

    while iterations < max_iterations:
        iterations += 1
        current_value = float(current.evaluate())

        neighbor = get_random_valid_neighbor(
            current,
            max_attempts=max_neighbor_attempts,
        )

        if neighbor is None:
            sampling_failures += 1
            objective_history.append(current_value)
            termination_reason = "neighbor_sampling_failed"
            break

        neighbor_value = float(neighbor.evaluate())

        if is_strictly_better(neighbor_value, current_value):
            current = neighbor.copy()
            accepted_moves += 1
        else:
            # A non-improving sample is rejected; keep the current state.
            rejected_moves += 1

        # Record one objective value for every attempted iteration.
        objective_history.append(float(current.evaluate()))

    duration = perf_counter() - start_time

    return HillClimbingResult(
        initial_state=initial,
        final_state=current.copy(),
        objective_history=objective_history,
        iterations=iterations,
        accepted_moves=accepted_moves,
        duration_seconds=duration,
        termination_reason=termination_reason,
        metadata={
            "variant": "stochastic",
            "max_iterations": max_iterations,
            "max_neighbor_attempts": max_neighbor_attempts,
            "rejected_moves": rejected_moves,
            "neighbor_sampling_failures": sampling_failures,
        },
    )