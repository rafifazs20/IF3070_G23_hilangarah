import unittest
from unittest.mock import patch

from src.hill_climbing_helper import HillClimbingResult
from src.random_restart_hill_climbing import (
    random_restart_hill_climbing,
)


class TinyState:
    """Minimal deterministic state for testing restart orchestration."""

    def __init__(self, name, value, valid=True):
        self.name = name
        self.value = value
        self.valid = valid

    def copy(self):
        return TinyState(
            self.name,
            self.value,
            self.valid,
        )

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


def make_result(
    initial_state,
    final_value,
    history,
    iterations,
    accepted_moves,
):
    return HillClimbingResult(
        initial_state=initial_state.copy(),
        final_state=TinyState(
            f"final-{initial_state.name}",
            final_value,
        ),
        objective_history=history,
        iterations=iterations,
        accepted_moves=accepted_moves,
        duration_seconds=0.01,
        termination_reason="local_optimum",
    )


class RandomRestartHillClimbingTests(unittest.TestCase):
    def setUp(self):
        self.ship = object()
        self.vehicles = {"A": object()}

        self.initial_states = [
            TinyState("start-1", 10),
            TinyState("start-2", 9),
            TinyState("start-3", 15),
        ]

        self.results = [
            make_result(
                self.initial_states[0],
                final_value=12,
                history=[10.0, 11.0, 12.0],
                iterations=2,
                accepted_moves=2,
            ),
            make_result(
                self.initial_states[1],
                final_value=9,
                history=[9.0, 9.0],
                iterations=1,
                accepted_moves=0,
            ),
            make_result(
                self.initial_states[2],
                final_value=16,
                history=[15.0, 16.0],
                iterations=1,
                accepted_moves=1,
            ),
        ]

    def run_with_mocks(self, max_restarts=3):
        with patch(
            "src.random_restart_hill_climbing."
            "CargoState.generate_random_initial_state",
            side_effect=self.initial_states,
        ) as generator:
            with patch(
                "src.random_restart_hill_climbing."
                "steepest_ascent_hill_climbing",
                side_effect=self.results,
            ) as search:
                result = random_restart_hill_climbing(
                    ship=self.ship,
                    vehicles=self.vehicles,
                    max_restarts=max_restarts,
                    max_iterations_per_restart=10,
                )

        return result, generator, search

    def test_returns_best_final_state(self):
        result, _, _ = self.run_with_mocks()

        self.assertEqual(
            result.final_state.evaluate(),
            16.0,
        )
        self.assertEqual(
            result.metadata["best_restart_number"],
            3,
        )

    def test_runs_exact_number_of_restarts(self):
        result, generator, search = self.run_with_mocks(
            max_restarts=3,
        )

        self.assertEqual(generator.call_count, 3)
        self.assertEqual(search.call_count, 3)
        self.assertEqual(
            result.metadata["restarts_completed"],
            3,
        )

    def test_aggregates_iteration_and_move_counts(self):
        result, _, _ = self.run_with_mocks()

        self.assertEqual(result.iterations, 4)
        self.assertEqual(result.accepted_moves, 3)
        self.assertEqual(
            result.metadata["iterations_per_restart"],
            [2, 1, 1],
        )
        self.assertEqual(
            result.metadata["total_iterations"],
            4,
        )

    def test_objective_history_tracks_best_seen_value(self):
        result, _, _ = self.run_with_mocks()

        self.assertEqual(
            result.objective_history,
            [10.0, 11.0, 12.0, 12.0, 12.0, 15.0, 16.0],
        )

        self.assertEqual(
            result.metadata["objective_history_iterations"],
            [0, 1, 2, 2, 3, 3, 4],
        )

        self.assertTrue(
            all(
                later >= earlier
                for earlier, later in zip(
                    result.objective_history,
                    result.objective_history[1:],
                )
            )
        )

        self.assertEqual(
            len(result.objective_history),
            len(result.metadata["objective_history_iterations"]),
        )

    def test_first_initial_state_and_best_final_state(self):
        result, _, _ = self.run_with_mocks()

        self.assertEqual(result.initial_state.name, "start-1")
        self.assertEqual(result.initial_state.evaluate(), 10.0)
        self.assertEqual(result.final_state.evaluate(), 16.0)

    def test_metadata_contains_each_restart(self):
        result, _, _ = self.run_with_mocks()

        summaries = result.metadata["restart_results"]

        self.assertEqual(len(summaries), 3)
        self.assertEqual(
            [
                item["final_objective"]
                for item in summaries
            ],
            [12.0, 9.0, 16.0],
        )

    def test_rejects_zero_restarts(self):
        with self.assertRaises(ValueError):
            random_restart_hill_climbing(
                self.ship,
                self.vehicles,
                max_restarts=0,
            )

    def test_rejects_zero_iterations_per_restart(self):
        with self.assertRaises(ValueError):
            random_restart_hill_climbing(
                self.ship,
                self.vehicles,
                max_iterations_per_restart=0,
            )

    def test_rejects_invalid_generated_initial_state(self):
        invalid_state = TinyState(
            "invalid",
            10,
            valid=False,
        )

        with patch(
            "src.random_restart_hill_climbing."
            "CargoState.generate_random_initial_state",
            return_value=invalid_state,
        ):
            with patch(
                "src.random_restart_hill_climbing."
                "steepest_ascent_hill_climbing",
            ) as search:
                with self.assertRaises(ValueError):
                    random_restart_hill_climbing(
                        self.ship,
                        self.vehicles,
                        max_restarts=1,
                    )

                search.assert_not_called()


if __name__ == "__main__":
    unittest.main()