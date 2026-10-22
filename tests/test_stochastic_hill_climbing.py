import unittest
from unittest.mock import patch

from src.stochastic_hill_climbing import stochastic_hill_climbing


class TinyState:
    """A minimal state for deterministic algorithm tests."""

    def __init__(self, name, value, valid=True):
        self.name = name
        self.value = value
        self.valid = valid

    def copy(self):
        return TinyState(self.name, self.value, self.valid)

    def is_valid(self):
        return self.valid

    def evaluate(self):
        return float(self.value)

    def to_dict(self):
        return {
            "name": self.name,
            "objective": self.value,
            "is_valid": self.valid,
        }


class StochasticHillClimbingTests(unittest.TestCase):
    def run_with_mocked_neighbors(
        self,
        initial,
        sampled_neighbors,
        max_iterations=3,
        max_neighbor_attempts=10,
    ):
        with patch(
            "src.stochastic_hill_climbing.get_random_valid_neighbor",
            side_effect=sampled_neighbors,
        ):
            return stochastic_hill_climbing(
                initial,
                max_iterations=max_iterations,
                max_neighbor_attempts=max_neighbor_attempts,
            )

    def test_accepts_improving_neighbor(self):
        initial = TinyState("A", 10)
        better = TinyState("B", 15)

        result = self.run_with_mocked_neighbors(
            initial,
            [better],
            max_iterations=1,
        )

        self.assertEqual(result.final_state.name, "B")
        self.assertEqual(result.final_state.evaluate(), 15.0)
        self.assertEqual(result.accepted_moves, 1)
        self.assertEqual(result.objective_history, [10.0, 15.0])

    def test_rejects_equal_and_worse_neighbors(self):
        initial = TinyState("A", 10)
        worse = TinyState("B", 8)
        equal = TinyState("C", 10)

        result = self.run_with_mocked_neighbors(
            initial,
            [worse, equal],
            max_iterations=2,
        )

        self.assertEqual(result.final_state.name, "A")
        self.assertEqual(result.final_state.evaluate(), 10.0)
        self.assertEqual(result.accepted_moves, 0)
        self.assertEqual(result.metadata["rejected_moves"], 2)
        self.assertEqual(
            result.objective_history,
            [10.0, 10.0, 10.0],
        )

    def test_stops_when_neighbor_sampling_fails(self):
        initial = TinyState("A", 10)

        result = self.run_with_mocked_neighbors(
            initial,
            [None],
            max_iterations=5,
        )

        self.assertEqual(result.final_state.name, "A")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(
            result.termination_reason,
            "neighbor_sampling_failed",
        )
        self.assertEqual(
            result.metadata["neighbor_sampling_failures"],
            1,
        )

    def test_rejects_invalid_initial_state(self):
        initial = TinyState("A", 10, valid=False)

        with self.assertRaises(ValueError):
            self.run_with_mocked_neighbors(initial, [])

    def test_rejects_negative_iteration_limit(self):
        initial = TinyState("A", 10)

        with self.assertRaises(ValueError):
            stochastic_hill_climbing(
                initial,
                max_iterations=-1,
            )

    def test_rejects_invalid_sampling_attempt_limit(self):
        initial = TinyState("A", 10)

        with self.assertRaises(ValueError):
            stochastic_hill_climbing(
                initial,
                max_neighbor_attempts=0,
            )

    def test_zero_iterations_returns_initial_state(self):
        initial = TinyState("A", 10)

        result = self.run_with_mocked_neighbors(
            initial,
            [],
            max_iterations=0,
        )

        self.assertEqual(result.final_state.name, "A")
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.objective_history, [10.0])
        self.assertEqual(
            result.termination_reason,
            "max_iterations",
        )

    def test_initial_state_is_not_modified(self):
        initial = TinyState("A", 10)
        better = TinyState("B", 20)

        self.run_with_mocked_neighbors(
            initial,
            [better],
            max_iterations=1,
        )

        self.assertEqual(initial.name, "A")
        self.assertEqual(initial.evaluate(), 10.0)

    def test_objective_history_never_decreases(self):
        initial = TinyState("A", 10)
        better = TinyState("B", 12)
        worse = TinyState("C", 9)
        best = TinyState("D", 15)

        result = self.run_with_mocked_neighbors(
            initial,
            [better, worse, best],
            max_iterations=3,
        )

        history = result.objective_history

        self.assertTrue(
            all(
                later >= earlier
                for earlier, later in zip(history, history[1:])
            )
        )
        self.assertEqual(history, [10.0, 12.0, 12.0, 15.0])


if __name__ == "__main__":
    unittest.main()