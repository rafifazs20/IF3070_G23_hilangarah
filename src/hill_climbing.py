from dataclasses import dataclass
from math import isclose
from random import choice
from time import perf_counter
from typing import Any


@dataclass
class HillClimbingResult:
    initial_state: Any
    final_state: Any
    objective_history: list[float]
    iterations: int
    accepted_moves: int
    duration_seconds: float
    termination_reason: str

    def to_dict(self) -> dict:
        return {
            "initial_state": self.initial_state.to_dict(),
            "final_state": self.final_state.to_dict(),
            "final_objective": self.final_state.evaluate(),
            "objective_history": self.objective_history,
            "iterations": self.iterations,
            "accepted_moves": self.accepted_moves,
            "duration_seconds": self.duration_seconds,
            "termination_reason": self.termination_reason,
        }


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

        neighbors = current.get_all_neighbors()
        valid_neighbors = [
            neighbor
            for neighbor in neighbors
            if neighbor.is_valid()
        ]

        if not valid_neighbors:
            objective_history.append(current_value)
            termination_reason = "no_valid_successors"
            break

        best_value = max(
            float(neighbor.evaluate())
            for neighbor in valid_neighbors
        )

        if best_value <= current_value or isclose(
            best_value,
            current_value,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            objective_history.append(current_value)
            termination_reason = "local_optimum"
            break

        best_neighbors = [
            neighbor
            for neighbor in valid_neighbors
            if isclose(
                float(neighbor.evaluate()),
                best_value,
                rel_tol=0.0,
                abs_tol=1e-9,
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