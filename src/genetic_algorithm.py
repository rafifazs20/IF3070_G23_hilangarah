from typing import List, Dict, Tuple
from src.models import CargoState, Ship, Vehicle

class GeneticAlgorithm:
    def __init__(
        self,
        ship: Ship,
        vehicles: Dict[str, Vehicle],
        pop_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        max_generations: int = 100,
        use_bonus_objective: bool = False
    ):
        self.ship = ship
        self.vehicles = vehicles
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.use_bonus_objective = use_bonus_objective

        self.history_max_fitness: List[float] = []
        self.history_avg_fitness: List[float] = []

    def _initialize_population(self) -> List[CargoState]:
        population: List[CargoState] = []
        for _ in range(self.pop_size):
            state = CargoState.generate_random_initial_state(
                ship=self.ship, 
                vehicles=self.vehicles, 
                use_bonus_objective=self.use_bonus_objective
            )
            population.append(state)
            
        return population