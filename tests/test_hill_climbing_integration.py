"""Integration tests using the project's real CargoState implementation."""
import random
import unittest

from src.hill_climbing_sideways import hill_climbing_with_sideways_move
from src.models import CargoState, Ship, Vehicle, VehiclePlacement
from src.random_restart_hill_climbing import random_restart_hill_climbing
from src.stochastic_hill_climbing import stochastic_hill_climbing
from src.steepest_ascent_hill_climbing import steepest_ascent_hill_climbing


class HillClimbingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.ship = Ship(w=4, l=4, max_capacity=10, id="Integration-Deck")
        self.vehicles = {
            "A": Vehicle("A", 1, 1, 10, 2, 1),
            "B": Vehicle("B", 1, 2, 20, 3, 2),
            "C": Vehicle("C", 2, 1, 15, 2, 3),
        }
        self.initial = CargoState(
            self.ship,
            self.vehicles,
            {
                "A": VehiclePlacement("A", True, 0, 0, "Horizontal"),
                "B": VehiclePlacement("B", True, 1, 0, "Horizontal"),
                "C": VehiclePlacement("C", False),
            },
        )
        self.assertTrue(self.initial.is_valid())

    def assert_valid_monotonic_result(self, result, expected_variant):
        self.assertIsInstance(result.initial_state, CargoState)
        self.assertIsInstance(result.final_state, CargoState)
        self.assertTrue(result.initial_state.is_valid())
        self.assertTrue(result.final_state.is_valid())
        self.assertTrue(result.final_state.get_total_weight() <= self.ship.max_capacity)
        self.assertTrue(result.objective_history)
        self.assertAlmostEqual(
            result.objective_history[0], result.initial_state.evaluate()
        )
        self.assertAlmostEqual(
            result.objective_history[-1], result.final_state.evaluate()
        )
        self.assertTrue(
            all(
                later + 1e-9 >= earlier
                for earlier, later in zip(
                    result.objective_history,
                    result.objective_history[1:],
                )
            ),
            msg=f"Objective history turun pada varian {expected_variant}",
        )
        self.assertGreaterEqual(result.iterations, 0)
        self.assertGreaterEqual(result.accepted_moves, 0)
        self.assertGreaterEqual(result.duration_seconds, 0.0)

    def test_steepest_ascent_on_real_cargo_state(self):
        random.seed(811)
        original = self.initial.to_dict()
        result = steepest_ascent_hill_climbing(
            self.initial,
            max_iterations=20,
        )
        self.assert_valid_monotonic_result(result, "steepest_ascent")
        self.assertEqual(len(result.objective_history), result.iterations + 1)
        self.assertEqual(self.initial.to_dict(), original)

    def test_sideways_move_on_real_cargo_state(self):
        random.seed(812)
        original = self.initial.to_dict()
        result = hill_climbing_with_sideways_move(
            self.initial,
            max_iterations=20,
            max_sideways_moves=3,
        )
        self.assert_valid_monotonic_result(result, "sideways_move")
        self.assertEqual(len(result.objective_history), result.iterations + 1)
        self.assertIn("total_sideways_moves", result.metadata)
        self.assertLessEqual(result.metadata["consecutive_sideways_moves_at_end"], 3)
        self.assertEqual(self.initial.to_dict(), original)

    def test_stochastic_on_real_cargo_state(self):
        random.seed(813)
        original = self.initial.to_dict()
        result = stochastic_hill_climbing(
            self.initial,
            max_iterations=20,
            max_neighbor_attempts=20,
        )
        self.assert_valid_monotonic_result(result, "stochastic")
        self.assertEqual(len(result.objective_history), result.iterations + 1)
        self.assertIn("rejected_moves", result.metadata)
        self.assertEqual(self.initial.to_dict(), original)

    def test_random_restart_on_real_cargo_state(self):
        random.seed(814)
        result = random_restart_hill_climbing(
            self.ship,
            self.vehicles,
            max_restarts=3,
            max_iterations_per_restart=20,
        )
        self.assert_valid_monotonic_result(result, "random_restart")
        self.assertEqual(result.metadata["restarts_completed"], 3)
        self.assertEqual(len(result.metadata["iterations_per_restart"]), 3)
        self.assertEqual(
            len(result.objective_history),
            result.iterations + result.metadata["restarts_completed"],
        )
        x_values = result.metadata["objective_history_iterations"]
        self.assertEqual(len(x_values), len(result.objective_history))
        self.assertEqual(x_values[0], 0)
        self.assertEqual(x_values[-1], result.iterations)
        self.assertEqual(x_values, sorted(x_values))
        self.assertAlmostEqual(
            result.final_state.evaluate(),
            max(result.metadata["best_objective_per_restart"]),
        )


if __name__ == "__main__":
    unittest.main()