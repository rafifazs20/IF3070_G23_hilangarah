
from random import choice
from time import perf_counter
from typing import Any

from .hill_climbing_helper import (
    HillClimbingResult,
    get_valid_neighbors,
    is_equal_objective,
    is_strictly_better,
)


def steepest_ascent_hill_climbing(
    initial_state: Any,
    max_iterations: int = 1000,
) -> HillClimbingResult:
    if max_iterations < 0:
        raise ValueError("max_iterations tidak boleh negatif.")

    start_time = perf_counter()

    initial = initial_state.copy()

    if not initial.is_valid():
        raise ValueError("Initial state harus memenuhi seluruh constraint.")

    current = initial.copy()
    objective_history = [float(current.evaluate())]

    iterations = 0
    accepted_moves = 0
    termination_reason = "max_iterations"

    while iterations < max_iterations:
        iterations += 1
        current_value = float(current.evaluate())

        valid_neighbors = get_valid_neighbors(current)

        if not valid_neighbors:
            objective_history.append(current_value)
            termination_reason = "no_valid_successors"
            break

        best_value = max(
            float(neighbor.evaluate())
            for neighbor in valid_neighbors
        )

        if not is_strictly_better(best_value, current_value):
            objective_history.append(current_value)
            termination_reason = "local_optimum"
            break

        best_neighbors = [
            neighbor
            for neighbor in valid_neighbors
            if is_equal_objective(
                float(neighbor.evaluate()),
                best_value,
            )
        ]

        current = choice(best_neighbors)
        accepted_moves += 1
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
    )