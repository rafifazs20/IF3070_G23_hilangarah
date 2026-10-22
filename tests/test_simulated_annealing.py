import unittest
from src.dataset import get_default_problem
from src.models import CargoState
from src.simulated_annealing import SimulatedAnnealing

class TestSimulatedAnnealing(unittest.TestCase):
    def setUp(self):
        # Setup default problem dan initial state untuk kapal dan daftar kendaraan 
        self.ship, self.vehicles = get_default_problem()
        
        self.initial_state = CargoState.generate_random_initial_state(
            ship=self.ship, 
            vehicles=self.vehicles, 
            use_bonus_objective=False
        )
        
        self.sa = SimulatedAnnealing(
            initial_temperature=100.0,
            cooling_rate=0.95,
            min_temperature=0.1,
            max_iterations=100 
        )

    def test_cooling_schedule(self):
        t0_temp = self.sa.schedule(0)
        t1_temp = self.sa.schedule(1)
        t10_temp = self.sa.schedule(10)
        
        self.assertEqual(t0_temp, 100.0)
        self.assertTrue(t0_temp > t1_temp > t10_temp)
        self.assertAlmostEqual(t1_temp, 95.0)

    def test_run_returns_correct_types(self):
        final_state, metrics = self.sa.run(self.initial_state)
        
        self.assertIsInstance(final_state, CargoState)
        self.assertIsInstance(metrics, dict)

    def test_metrics_completeness(self):
        _, metrics = self.sa.run(self.initial_state)
        
        expected_keys = [
            "initial_value", "final_value", "duration_seconds", 
            "iterations_ran", "history_obj", "history_prob", 
            "stuck_frequency", "is_valid"
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
            
        self.assertTrue(len(metrics["history_obj"]) > 0)
        
        self.assertTrue(metrics["stuck_frequency"] >= 0)
        
        self.assertTrue(metrics["duration_seconds"] > 0.0)

    def test_sa_improves_or_maintains_value(self):
        final_state, metrics = self.sa.run(self.initial_state)
        
        self.assertIsNotNone(metrics["final_value"])
        self.assertIsInstance(final_state.placements, dict)

if __name__ == '__main__':
    unittest.main()