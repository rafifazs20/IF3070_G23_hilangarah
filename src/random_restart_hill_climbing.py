
from time import perf_counter
from typing import Dict

from .models import CargoState, Ship, Vehicle
from .hill_climbing_helper import HillClimbingResult
from .steepest_ascent_hill_climbing import (
    steepest_ascent_hill_climbing,
)


def random_restart_hill_climbing(
    ship: Ship,
    vehicles: Dict[str, Vehicle],
    max_restarts: int = 10,
    max_iterations_per_restart: int = 1000,
    use_bonus_objective: bool = False,
) -> HillClimbingResult:
    """
    Run multiple independent Steepest Ascent searches from random initial
    states and return the best final state found.

    max_restarts is the total number of independent attempts, including
    the first attempt.
    """
    if max_restarts < 1:
        raise ValueError("max_restarts harus minimal 1.")

    if max_iterations_per_restart < 1:
        raise ValueError(
            "max_iterations_per_restart harus minimal 1."
        )

    start_time = perf_counter()
    restart_results = []

    for restart_number in range(1, max_restarts + 1):
        initial_state = CargoState.generate_random_initial_state(
            ship=ship,
            vehicles=vehicles,
            use_bonus_objective=use_bonus_objective,
        )

        if not initial_state.is_valid():
            raise ValueError(
                f"Initial state pada restart {restart_number} "
                "tidak memenuhi seluruh constraint."
            )

        result = steepest_ascent_hill_climbing(
            initial_state=initial_state,
            max_iterations=max_iterations_per_restart,
        )

        if not result.final_state.is_valid():
            raise RuntimeError(
                f"Final state pada restart {restart_number} "
                "tidak valid."
            )

        restart_results.append(result)

    # Select the run that achieved the highest final objective.
    best_restart_index = max(
        range(len(restart_results)),
        key=lambda index: restart_results[index].final_state.evaluate(),
    )

    best_result = restart_results[best_restart_index]

    total_iterations = sum(
        result.iterations for result in restart_results
    )

    total_accepted_moves = sum(
        result.accepted_moves for result in restart_results
    )

    # Aggregate the best objective found so far after each search iteration.
    # This trace remains non-decreasing across restarts.
    best_seen = float(restart_results[0].objective_history[0])
    aggregate_history = [best_seen]

    for result in restart_results:
        for value in result.objective_history[1:]:
            best_seen = max(best_seen, float(value))
            aggregate_history.append(best_seen)

    restart_summaries = []

    for restart_number, result in enumerate(
        restart_results,
        start=1,
    ):
        restart_summaries.append({
            "restart_number": restart_number,
            "initial_state": result.initial_state.to_dict(),
            "final_state": result.final_state.to_dict(),
            "initial_objective": float(
                result.initial_state.evaluate()
            ),
            "final_objective": float(
                result.final_state.evaluate()
            ),
            "objective_history": result.objective_history,
            "iterations": result.iterations,
            "accepted_moves": result.accepted_moves,
            "duration_seconds": result.duration_seconds,
            "termination_reason": result.termination_reason,
        })

    duration = perf_counter() - start_time

    return HillClimbingResult(
        initial_state=restart_results[0].initial_state.copy(),
        final_state=best_result.final_state.copy(),
        objective_history=aggregate_history,
        iterations=total_iterations,
        accepted_moves=total_accepted_moves,
        duration_seconds=duration,
        termination_reason="max_restarts_reached",
        metadata={
            "variant": "random_restart",
            "max_restarts": max_restarts,
            "restarts_completed": len(restart_results),
            "max_iterations_per_restart": (
                max_iterations_per_restart
            ),
            "iterations_per_restart": [
                result.iterations for result in restart_results
            ],
            "best_restart_number": best_restart_index + 1,
            "best_objective_per_restart": [
                float(result.final_state.evaluate())
                for result in restart_results
            ],
            "restart_results": restart_summaries,
            "total_iterations": total_iterations,
        },
    )