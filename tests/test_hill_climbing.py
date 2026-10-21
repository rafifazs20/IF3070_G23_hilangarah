import unittest

from src.models import CargoState, Ship, Vehicle, VehiclePlacement
from src.hill_climbing import steepest_ascent_hill_climbing


class HillClimbingTests(unittest.TestCase):
    def setUp(self):
        self.ship = Ship(w=4, l=4, max_capacity=10)

        self.vehicles = {
            "A": Vehicle(
                id="A",
                w=2,
                l=2,
                shipping_fee=60,
                weight=4,
                eta=2,
            ),
            "B": Vehicle(
                id="B",
                w=1,
                l=2,
                shipping_fee=30,
                weight=3,
                eta=3,
            ),
        }

    def make_state(self, placements, ship=None, vehicles=None):
        return CargoState(
            ship=ship or self.ship,
            vehicles=vehicles or self.vehicles,
            placements=placements,
        )

    def test_valid_state_and_objective(self):
        state = self.make_state({
            "A": VehiclePlacement("A", True, 0, 0, "Horizontal"),
            "B": VehiclePlacement("B", True, 2, 0, "Horizontal"),
        })

        self.assertTrue(state.is_valid())
        self.assertEqual(state.evaluate(), 90.0)

    def test_overlap_is_invalid(self):
        state = self.make_state({
            "A": VehiclePlacement("A", True, 0, 0, "Horizontal"),
            "B": VehiclePlacement("B", True, 1, 1, "Horizontal"),
        })

        self.assertFalse(state.is_valid())

    def test_out_of_bounds_is_invalid(self):
        state = self.make_state({
            "A": VehiclePlacement("A", True, 3, 0, "Horizontal"),
            "B": VehiclePlacement("B", False),
        })

        self.assertFalse(state.is_valid())

    def test_excess_weight_is_invalid(self):
        vehicles = {
            "A": Vehicle("A", 1, 1, 60, 6, 2),
            "B": Vehicle("B", 1, 1, 30, 6, 3),
        }

        state = self.make_state(
            {
                "A": VehiclePlacement("A", True, 0, 0),
                "B": VehiclePlacement("B", True, 2, 0),
            },
            vehicles=vehicles,
        )

        self.assertFalse(state.is_valid())

    def test_invalid_initial_state_is_rejected(self):
        state = self.make_state({
            "A": VehiclePlacement("A", True, 3, 0, "Horizontal"),
            "B": VehiclePlacement("B", False),
        })

        with self.assertRaises(ValueError):
            steepest_ascent_hill_climbing(state)

    def test_hill_climbing_reaches_local_optimum(self):
        ship = Ship(w=1, l=1, max_capacity=1)
        vehicles = {
            "A": Vehicle("A", 1, 1, 10, 1, 1),
        }

        initial = CargoState(
            ship,
            vehicles,
            {
                "A": VehiclePlacement("A", False),
            },
        )

        result = steepest_ascent_hill_climbing(initial)

        self.assertTrue(result.initial_state.is_valid())
        self.assertTrue(result.final_state.is_valid())
        self.assertEqual(result.final_state.evaluate(), 10.0)
        self.assertEqual(result.termination_reason, "local_optimum")

        history = result.objective_history

        self.assertTrue(
            all(
                later >= earlier
                for earlier, later in zip(history, history[1:])
            )
        )

    def test_initial_state_is_not_modified(self):
        initial = CargoState.generate_random_initial_state(
            self.ship,
            self.vehicles,
        )
        original = initial.to_dict()

        steepest_ascent_hill_climbing(initial)

        self.assertEqual(initial.to_dict(), original)

    def test_final_state_has_no_better_valid_successor(self):
        initial = CargoState.generate_random_initial_state(
            self.ship,
            self.vehicles,
        )

        result = steepest_ascent_hill_climbing(initial)

        final_state = result.final_state
        final_value = final_state.evaluate()

        if result.termination_reason == "local_optimum":
            for neighbor in final_state.get_all_neighbors():
                if neighbor.is_valid():
                    self.assertLessEqual(
                        neighbor.evaluate(),
                        final_value,
                    )


if __name__ == "__main__":
    unittest.main()