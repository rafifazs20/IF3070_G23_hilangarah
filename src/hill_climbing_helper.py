from dataclasses import dataclass, field
from math import isclose
from typing import Any, List, Optional

OBJECTIVE_EPSILON = 1e-9


@dataclass
class HillClimbingResult:
    initial_state: Any
    final_state: Any
    objective_history: List[float]
    iterations: int
    accepted_moves: int
    duration_seconds: float
    termination_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_state": self.initial_state.to_dict(),
            "final_state": self.final_state.to_dict(),
            "initial_objective": float(self.initial_state.evaluate()),
            "final_objective": float(self.final_state.evaluate()),
            "objective_history": self.objective_history,
            "iterations": self.iterations,
            "accepted_moves": self.accepted_moves,
            "duration_seconds": self.duration_seconds,
            "termination_reason": self.termination_reason,
            "metadata": self.metadata.copy(),
        }


def is_strictly_better(
    candidate_value: float,
    current_value: float,
) -> bool:
    """Return True only when the candidate improves the objective."""
    return candidate_value > current_value + OBJECTIVE_EPSILON


def is_equal_objective(
    first_value: float,
    second_value: float,
) -> bool:
    """Compare objective values with a small numerical tolerance."""
    return isclose(
        first_value,
        second_value,
        rel_tol=0.0,
        abs_tol=OBJECTIVE_EPSILON,
    )


def get_valid_neighbors(state: Any) -> List[Any]:
    """Return all neighbors that satisfy the problem constraints."""
    return [
        neighbor
        for neighbor in state.get_all_neighbors()
        if neighbor.is_valid()
    ]


def get_random_valid_neighbor(
    state: Any,
    max_attempts: int = 100,
) -> Optional[Any]:
    """Sample a random valid neighbor, returning None if sampling fails."""
    if max_attempts < 1:
        raise ValueError("max_attempts harus minimal 1.")

    for _ in range(max_attempts):
        neighbor = state.get_random_neighbor()

        if neighbor is not None and neighbor.is_valid():
            return neighbor

    return None