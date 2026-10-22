import unittest
from unittest.mock import patch

from src.models import VehiclePlacement
from src.hill_climbing_sideways import (
    hill_climbing_with_sideways_move,
)


class TinyState:
    """Small deterministic state for testing HC control flow."""

    def __init__(self, name, value, x, valid=True):
        self.name = name
        self.value = value
        self.valid = valid
        self.placements = {
            "A": VehiclePlacement(
                "A",
                True,
                x,
                0,
                "Horizontal",
            )
        }
        self.neighbors = []

    def copy(self):
        cloned = TinyState(
            self.name,
            self.value,
            self.placements["A"].x,
            self.valid,
        )
        cloned.neighbors = list(self.neighbors)
        return cloned

    def is_valid(self):
        return self.valid

    def evaluate(self):
        return float(self.value)

    def to_dict(self):
        return {
            "name": self.name,
            "objective": self.value,
        }


class SidewaysHillClimbingTests(unittest.TestCase):
    def run_with_mocked_neighbors(
        self,
        initial,
        max_iterations=10,
        max_sideways_moves=1,
    ):
        with patch(
            "src.hill_climbing_sideways.get_valid_neighbors",
            side_effect=lambda state: state.neighbors,
        ):
            return hill_climbing_with_sideways_move(
                initial,
                max_iterations=max_iterations,
                max_sideways_moves=max_sideways_moves,
            )

    def test_accepts_sideways_move(self):
        a = TinyState("A", 10, 0)
        b = TinyState("B", 10, 1)

        a.neighbors = [b]
        b.neighbors = [a]

        result = self.run_with_mocked_neighbors(a)

        self.assertEqual(result.final_state.name, "B")
        self.assertEqual(result.accepted_moves, 1)
        self.assertEqual(
            result.metadata["total_sideways_moves"],
            1,
        )
        self.assertEqual(
            result.termination_reason,
            "local_optimum",
        )

    def test_prefers_improvement_over_sideways(self):
        a = TinyState("A", 10, 0)
        b = TinyState("B", 10, 1)
        c = TinyState("C", 12, 2)

        a.neighbors = [b, c]
        b.neighbors = [a]
        c.neighbors = []

        result = self.run_with_mocked_neighbors(a)

        self.assertEqual(result.final_state.name, "C")
        self.assertEqual(result.final_state.evaluate(), 12.0)
        self.assertEqual(
            result.metadata["total_sideways_moves"],
            0,
        )

    def test_stops_when_sideways_limit_is_reached(self):
        a = TinyState("A", 10, 0)
        b = TinyState("B", 10, 1)
        c = TinyState("C", 10, 2)

        a.neighbors = [b]
        b.neighbors = [a, c]
        c.neighbors = [b]

        result = self.run_with_mocked_neighbors(
            a,
            max_sideways_moves=1,
        )

        self.assertEqual(result.final_state.name, "B")
        self.assertEqual(result.accepted_moves, 1)
        self.assertEqual(
            result.metadata["total_sideways_moves"],
            1,
        )
        self.assertEqual(
            result.termination_reason,
            "sideways_limit_reached",
        )

    def test_zero_sideways_limit_disallows_sideways(self):
        a = TinyState("A", 10, 0)
        b = TinyState("B", 10, 1)

        a.neighbors = [b]

        result = self.run_with_mocked_neighbors(
            a,
            max_sideways_moves=0,
        )

        self.assertEqual(result.final_state.name, "A")
        self.assertEqual(result.accepted_moves, 0)
        self.assertEqual(
            result.termination_reason,
            "sideways_limit_reached",
        )

    def test_rejects_invalid_initial_state(self):
        initial = TinyState("A", 10, 0, valid=False)

        with self.assertRaises(ValueError):
            self.run_with_mocked_neighbors(initial)

    def test_rejects_negative_sideways_limit(self):
        initial = TinyState("A", 10, 0)

        with self.assertRaises(ValueError):
            hill_climbing_with_sideways_move(
                initial,
                max_sideways_moves=-1,
            )

    def test_initial_state_is_not_modified(self):
        a = TinyState("A", 10, 0)
        b = TinyState("B", 10, 1)

        a.neighbors = [b]
        b.neighbors = [a]

        original_x = a.placements["A"].x
        original_value = a.evaluate()

        self.run_with_mocked_neighbors(a)

        self.assertEqual(a.placements["A"].x, original_x)
        self.assertEqual(a.evaluate(), original_value)


if __name__ == "__main__":
    unittest.main()