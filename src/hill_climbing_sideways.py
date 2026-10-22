
from random import choice
from time import perf_counter
from typing import Any, List, Tuple

from .hill_climbing_helper import (
    HillClimbingResult,
    get_valid_neighbors,
    is_equal_objective,
    is_strictly_better,
)


def _state_signature(state: Any) -> Tuple:
    """
    Create a signature of a cargo configuration.

    Orientation and coordinates are ignored for vehicles outside the ship,
    because they do not currently affect the loaded configuration.
    """
    return tuple(
        (
            vehicle_id,
            placement.in_ship,
            placement.x if placement.in_ship else None,
            placement.y if placement.in_ship else None,
            placement.orientation if placement.in_ship else None,
        )
        for vehicle_id, placement in sorted(state.placements.items())
    )


def hill_climbing_with_sideways_move(
    initial_state: Any,
    max_iterations: int = 1000,
    max_sideways_moves: int = 100,
) -> HillClimbingResult:
    if max_iterations < 0:
        raise ValueError("max_iterations tidak boleh negatif.")

    if max_sideways_moves < 0:
        raise ValueError("max_sideways_moves tidak boleh negatif.")

    start_time = perf_counter()

    initial = initial_state.copy()

    if not initial.is_valid():
        raise ValueError("Initial state harus memenuhi seluruh constraint.")

    current = initial.copy()
    objective_history = [float(current.evaluate())]

    iterations = 0
    accepted_moves = 0
    total_sideways_moves = 0
    consecutive_sideways_moves = 0
    sideways_move_iterations: List[int] = []

    visited = {_state_signature(current)}
    termination_reason = "max_iterations"

    while iterations < max_iterations:
        iterations += 1
        current_value = float(current.evaluate())

        valid_neighbors = get_valid_neighbors(current)

        if not valid_neighbors:
            objective_history.append(current_value)
            termination_reason = "no_valid_successors"
            break

        # Ignore previously visited configurations to avoid cycling.
        unique_neighbors = {}
        current_signature = _state_signature(current)

        for neighbor in valid_neighbors:
            signature = _state_signature(neighbor)

            if signature == current_signature or signature in visited:
                continue

            unique_neighbors.setdefault(signature, neighbor)

        candidates = list(unique_neighbors.values())

        # 1. Always prioritize an improving move when one exists.
        improving_neighbors = [
            neighbor
            for neighbor in candidates
            if is_strictly_better(
                float(neighbor.evaluate()),
                current_value,
            )
        ]

        if improving_neighbors:
            best_value = max(
                float(neighbor.evaluate())
                for neighbor in improving_neighbors
            )

            best_neighbors = [
                neighbor
                for neighbor in improving_neighbors
                if is_equal_objective(
                    float(neighbor.evaluate()),
                    best_value,
                )
            ]

            current = choice(best_neighbors)
            visited.add(_state_signature(current))

            accepted_moves += 1
            consecutive_sideways_moves = 0
            objective_history.append(float(current.evaluate()))
            continue

        # 2. If no improving move exists, consider sideways moves.
        equal_neighbors = [
            neighbor
            for neighbor in candidates
            if is_equal_objective(
                float(neighbor.evaluate()),
                current_value,
            )
        ]

        if not equal_neighbors:
            objective_history.append(current_value)
            termination_reason = "local_optimum"
            break

        if consecutive_sideways_moves >= max_sideways_moves:
            objective_history.append(current_value)
            termination_reason = "sideways_limit_reached"
            break

        current = choice(equal_neighbors)
        visited.add(_state_signature(current))

        accepted_moves += 1
        total_sideways_moves += 1
        consecutive_sideways_moves += 1
        sideways_move_iterations.append(iterations)

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
            "variant": "sideways",
            "max_sideways_moves": max_sideways_moves,
            "total_sideways_moves": total_sideways_moves,
            "consecutive_sideways_moves_at_end": (
                consecutive_sideways_moves
            ),
            "sideways_move_iterations": sideways_move_iterations,
        },
    )