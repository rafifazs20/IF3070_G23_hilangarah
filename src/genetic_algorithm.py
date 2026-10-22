import random
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

    # Penyesuaian CargoState Serialization punya Rafif
    def _calculate_fitness(self, state: CargoState) -> float:
        oob, ovl, ex_w = state.count_violations()
        base_fee = state.get_total_shipping_fee()

        if oob == 0 and ovl == 0 and ex_w == 0:
            if not self.use_bonus_objective:
                return float(base_fee)
            else:
                bonus_score = 0.0
                for vid, p in state.placements.items():
                    if p.in_ship:
                        v = self.vehicles[vid]
                        urgency_bonus = v.shipping_fee * (1.0 / max(1, v.eta)) * 0.1
                        bonus_score += v.shipping_fee + urgency_bonus
                return round(bonus_score, 4)
        else:
            penalty = (oob * 500.0) + (ovl * 500.0) + (ex_w * 50.0)
            return float(base_fee) - penalty

    def _roulette_wheel_selection(self, population: List[CargoState]) -> CargoState:
        fitnesses = [self._calculate_fitness(ind) for ind in population]
        min_fit = min(fitnesses)
        
        shifted_fitnesses = [f - min_fit + 1.0 for f in fitnesses]
        
        # Pilih individu, semakin besar fitness semakin besar persentase
        selected = random.choices(population, weights=shifted_fitnesses, k=1)[0]
        
        return selected.copy()

    def _crossover(self, parent1: CargoState, parent2: CargoState) -> Tuple[CargoState, CargoState]:
        # Cek masuk atau tidak
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        # Crossover
        for vid in self.vehicles.keys():
            if random.random() < 0.5:
                child1.placements[vid] = parent2.placements[vid].copy()
                child2.placements[vid] = parent1.placements[vid].copy()

        child1.invalidate_cache()
        child2.invalidate_cache()
        
        return child1, child2

    def _mutate(self, state: CargoState) -> CargoState:
        # Random untuk menentukan mutasi
        if random.random() < self.mutation_rate:
            mutated_state = state.get_random_neighbor(max_attempts=15)
            
            if mutated_state is not None:
                return mutated_state
                
        return state

    def run(self) -> Tuple[CargoState, List[float], List[float]]:
        population = self._initialize_population()
        best_state = max(population, key=self._calculate_fitness)
        self.initial_best_state = best_state.copy()
        
        for generation in range(self.max_generations):
            # Evaluasi populasi saat ini
            fitnesses = [self._calculate_fitness(ind) for ind in population]
            gen_max_fit = max(fitnesses)
            gen_avg_fit = sum(fitnesses) / self.pop_size
            
            self.history_max_fitness.append(gen_max_fit)
            self.history_avg_fitness.append(gen_avg_fit)

            # Cek apa ada yang mengalahkan best
            current_best = max(population, key=self._calculate_fitness)
            if self._calculate_fitness(current_best) > self._calculate_fitness(best_state):
                best_state = current_best.copy()

            sorted_pop = sorted(population, key=self._calculate_fitness, reverse=True)
            new_population = [sorted_pop[0].copy(), sorted_pop[1].copy()]

            # Inti GA (Seleksi, Crossover, Mutasi)
            while len(new_population) < self.pop_size:
                p1 = self._roulette_wheel_selection(population)
                p2 = self._roulette_wheel_selection(population)
                
                c1, c2 = self._crossover(p1, p2)
                
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)

            population = new_population

            if (generation + 1) % 10 == 0:
                print(
                    f"  Generasi {generation + 1}/{self.max_generations} | "
                    f"fitness maks: {gen_max_fit:.2f} | "
                    f"fitness rata-rata: {gen_avg_fit:.2f}"
                )
            
        return best_state, self.history_max_fitness, self.history_avg_fitness
