from __future__ import annotations

import argparse
import random
from time import perf_counter
from typing import Any

from src.dataset import get_default_problem, load_problem_from_json
from src.genetic_algorithm import GeneticAlgorithm
from src.hill_climbing_sideways import hill_climbing_with_sideways_move
from src.hill_climbing_helper import HillClimbingResult
from src.models import CargoState
from src.random_restart_hill_climbing import random_restart_hill_climbing
from src.simulated_annealing import SimulatedAnnealing
from src.stochastic_hill_climbing import stochastic_hill_climbing
from src.steepest_ascent_hill_climbing import steepest_ascent_hill_climbing


ALGORITHMS = (
    "steepest-ascent",
    "sideways-move",
    "stochastic-hill-climbing",
    "random-restart",
    "simulated-annealing",
    "genetic-algorithm",
)


def run_hill_climbing(name: str, ship: Any, vehicles: dict, args: argparse.Namespace) -> HillClimbingResult:
    initial = CargoState.generate_random_initial_state(ship, vehicles)
    if name == "steepest-ascent":
        return steepest_ascent_hill_climbing(initial, max_iterations=args.max_iterations)
    if name == "sideways-move":
        return hill_climbing_with_sideways_move(
            initial, max_iterations=args.max_iterations,
            max_sideways_moves=args.max_sideways_moves,
        )
    if name == "stochastic-hill-climbing":
        return stochastic_hill_climbing(
            initial, max_iterations=args.max_iterations,
            max_neighbor_attempts=args.max_neighbor_attempts,
        )
    return random_restart_hill_climbing(
        ship, vehicles, max_restarts=args.max_restarts,
        max_iterations_per_restart=args.max_iterations,
    )


def run_simulated_annealing(ship: Any, vehicles: dict, args: argparse.Namespace) -> tuple[CargoState, dict]:
    initial = CargoState.generate_random_initial_state(ship, vehicles)
    algorithm = SimulatedAnnealing(
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        min_temperature=args.min_temperature,
        max_iterations=args.max_iterations,
    )
    return algorithm.run(initial)


def run_genetic_algorithm(ship: Any, vehicles: dict, args: argparse.Namespace) -> tuple[CargoState, list[float], list[float]]:
    algorithm = GeneticAlgorithm(
        ship=ship,
        vehicles=vehicles,
        pop_size=args.population_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        max_generations=args.generations,
    )
    return algorithm.run()


def print_result(name: str, run_number: int, initial: float, final: float,
                 iterations: int, duration: float, valid: bool,
                 state: CargoState | None = None) -> None:
    print(
        f"{name} run {run_number}: objective {initial:.2f} -> {final:.2f}, "
        f"iterasi={iterations}, durasi={duration:.3f}s, valid={valid}"
    )
    if state and len(state.ships) > 1:
        for ship in state.ships:
            onboard = [vid for vid, placement in state.placements.items()
                       if placement.in_ship and (placement.ship_id or state.ship.id) == ship.id]
            fee = sum(state.vehicles[vid].shipping_fee for vid in onboard)
            weight = state.get_total_weight(ship.id)
            print(f"  {ship.id}: {len(onboard)} kendaraan, fee={fee}, berat={weight}/{ship.max_capacity}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilih algoritma pencarian untuk masalah pemuatan kapal.")
    parser.add_argument("--algorithm", choices=ALGORITHMS, help="Jalankan pilihan langsung tanpa menu.")
    parser.add_argument("--data", help="Path dataset JSON; gunakan dataset bawaan jika tidak diisi.")
    parser.add_argument("--num-ships", type=int, default=1,
                        help="Jumlah kapal untuk dataset bawaan (minimal 1).")
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--repeats", type=int, default=3, help="Jumlah run Hill-Climbing dan Simulated Annealing.")
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--max-sideways-moves", type=int, default=20)
    parser.add_argument("--max-neighbor-attempts", type=int, default=20)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--initial-temperature", type=float, default=1000.0)
    parser.add_argument("--cooling-rate", type=float, default=0.99)
    parser.add_argument("--min-temperature", type=float, default=1e-3)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    args = parser.parse_args()

    if (args.repeats < 1 or args.max_iterations < 0 or args.population_size < 2
            or args.generations < 1 or args.max_sideways_moves < 0
            or args.max_neighbor_attempts < 1 or args.max_restarts < 1
            or args.num_ships < 1):
        parser.error("Periksa repeats, batas iterasi, konfigurasi Hill-Climbing, dan ukuran populasi.")
    if not 0 <= args.mutation_rate <= 1 or not 0 <= args.crossover_rate <= 1:
        parser.error("mutation-rate dan crossover-rate harus berada pada rentang 0 sampai 1")
    if args.initial_temperature <= 0 or args.min_temperature < 0 or not 0 < args.cooling_rate < 1:
        parser.error("parameter temperatur tidak valid")

    labels = (
        "Steepest Ascent Hill-Climbing",
        "Hill-Climbing with Sideways Move",
        "Stochastic Hill-Climbing",
        "Random Restart Hill-Climbing",
        "Simulated Annealing",
        "Genetic Algorithm",
    )
    ship, vehicles = load_problem_from_json(args.data) if args.data else get_default_problem(args.num_ships)
    seed_offset = 0

    while True:
        if args.algorithm:
            name = args.algorithm
        else:
            print("\nPilih algoritma:")
            for index, label in enumerate(labels, start=1):
                print(f"{index}. {label}")
            print("0. Keluar")
            choice = input("Pilihan: ").strip()
            if choice == "0":
                print("Selesai.")
                break
            if not choice.isdigit() or not 1 <= int(choice) <= len(ALGORITHMS):
                print("Pilihan tidak valid. Masukkan angka 0 sampai 6.")
                continue
            name = ALGORITHMS[int(choice) - 1]

        runs = 1 if name == "genetic-algorithm" else args.repeats
        for run_number in range(1, runs + 1):
            random.seed(args.seed + seed_offset)
            seed_offset += 1

            if name in ALGORITHMS[:4]:
                result = run_hill_climbing(name, ship, vehicles, args)
                print_result(name, run_number, result.initial_state.evaluate(), result.final_state.evaluate(),
                             result.iterations, result.duration_seconds, result.final_state.is_valid(),
                             result.final_state)
            elif name == "simulated-annealing":
                final, metrics = run_simulated_annealing(ship, vehicles, args)
                print_result(name, run_number, metrics["initial_value"], metrics["final_value"],
                             metrics["iterations_ran"], metrics["duration_seconds"], final.is_valid(), final)
            else:
                start = perf_counter()
                final, history_max, _ = run_genetic_algorithm(ship, vehicles, args)
                initial_value = history_max[0] if history_max else final.evaluate()
                print_result(name, run_number, initial_value, final.evaluate(), len(history_max),
                             perf_counter() - start, final.is_valid(), final)

        if args.algorithm:
            break


if __name__ == "__main__":
    main()
