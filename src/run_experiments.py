from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any

from src.dataset import get_default_problem, load_problem_from_json
from src.genetic_algorithm import GeneticAlgorithm
from src.hill_climbing_sideways import hill_climbing_with_sideways_move
from src.models import CargoState
from src.random_restart_hill_climbing import random_restart_hill_climbing
from src.simulated_annealing import SimulatedAnnealing
from src.stochastic_hill_climbing import stochastic_hill_climbing
from src.steepest_ascent_hill_climbing import steepest_ascent_hill_climbing
from src.visualizer import plot_deck, plot_experiment_results


HC_ALGORITHMS = (
    ("Steepest Ascent HC", steepest_ascent_hill_climbing),
    ("Sideways Move HC", hill_climbing_with_sideways_move),
    ("Stochastic HC", stochastic_hill_climbing),
)


def _run_hc(name: str, algorithm: Any, ship: Any, vehicles: dict, args: argparse.Namespace):
    if name == "Random Restart HC":
        return random_restart_hill_climbing(
            ship, vehicles, max_restarts=args.max_restarts,
            max_iterations_per_restart=args.max_iterations,
        )
    initial = CargoState.generate_random_initial_state(ship, vehicles)
    if name == "Sideways Move HC":
        return algorithm(initial, max_iterations=args.max_iterations,
                         max_sideways_moves=args.max_sideways_moves)
    if name == "Stochastic HC":
        return algorithm(initial, max_iterations=args.max_iterations,
                         max_neighbor_attempts=args.max_neighbor_attempts)
    return algorithm(initial, max_iterations=args.max_iterations)


def _record(name: str, run: int, seed: int, initial: CargoState, final: CargoState,
            history: list[float], iterations: int, duration: float,
            **configuration: Any) -> dict[str, Any]:
    return {
        "algorithm": name, "run": run, "seed": seed,
        "initial_objective": float(initial.evaluate()),
        "final_objective": float(final.evaluate()),
        "highest_objective": float(max(history, default=final.evaluate())),
        "initial_state": initial.to_dict(),
        "final_state": final.to_dict(),
        "objective_history": [float(value) for value in history],
        "iterations": iterations, "duration_seconds": duration,
        "final_valid": final.is_valid(), **configuration,
    }


def _print_run_progress(index: int, total: int, record: dict[str, Any], best: float) -> None:
    print(
        f"[{index:02d}/{total:02d}] {record['algorithm']:<24} "
        f"run {record['run']}: awal={record['initial_objective']:.2f}, "
        f"akhir={record['final_objective']:.2f}, "
        f"tertinggi={record['highest_objective']:.2f}, "
        f"tertinggi global={best:.2f}, "
        f"iterasi={record['iterations']}, "
        f"durasi={record['duration_seconds']:.2f}s"
    )


def run_experiments(args: argparse.Namespace) -> dict[str, Any]:
    ship, vehicles = load_problem_from_json(args.data) if args.data else get_default_problem()
    out_dir = Path(args.output_dir)
    image_dir = out_dir / "states"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    states: dict[str, tuple[CargoState, CargoState]] = {}
    seed = args.seed
    total_runs = 4 * 3 + 3 + len(args.population_sizes) * len(args.generation_counts)
    best_score = float("-inf")

    hc_variants = list(HC_ALGORITHMS) + [("Random Restart HC", None)]
    for name, algorithm in hc_variants:
        for run in range(1, 4):
            random.seed(seed)
            result = _run_hc(name, algorithm, ship, vehicles, args)
            record = _record(name, run, seed, result.initial_state, result.final_state,
                             result.objective_history, result.iterations,
                             result.duration_seconds, termination_reason=result.termination_reason)
            records.append(record)
            states[f"{_slug(name)}_run_{run}"] = (result.initial_state, result.final_state)
            best_score = max(best_score, record["highest_objective"])
            seed += 1
            _print_run_progress(len(records), total_runs, record, best_score)

    for run in range(1, 4):
        random.seed(seed)
        initial = CargoState.generate_random_initial_state(ship, vehicles)
        algorithm = SimulatedAnnealing(
            initial_temperature=args.initial_temperature,
            cooling_rate=args.cooling_rate,
            min_temperature=args.min_temperature,
            max_iterations=args.max_iterations,
        )
        final, metrics = algorithm.run(initial)
        record = _record("Simulated Annealing", run, seed, initial, final,
                         metrics["history_obj"], metrics["iterations_ran"],
                         metrics["duration_seconds"], stuck_frequency=metrics["stuck_frequency"])
        records.append(record)
        states[f"simulated_annealing_run_{run}"] = (initial, final)
        best_score = max(best_score, record["highest_objective"])
        seed += 1
        _print_run_progress(len(records), total_runs, record, best_score)

    ga_run = 0
    for population in args.population_sizes:
        for generations in args.generation_counts:
            random.seed(seed)
            algorithm = GeneticAlgorithm(
                ship, vehicles, pop_size=population,
                mutation_rate=args.mutation_rate, crossover_rate=args.crossover_rate,
                max_generations=generations,
            )
            started = perf_counter()
            final, max_history, avg_history = algorithm.run()
            duration = perf_counter() - started
            initial = algorithm.initial_best_state
            ga_run += 1
            record = _record("Genetic Algorithm", ga_run, seed, initial, final,
                             max_history, len(max_history), duration,
                             average_fitness_history=avg_history,
                             population_size=population, generations=generations)
            records.append(record)
            states[f"genetic_pop_{population}_gen_{generations}"] = (initial, final)
            best_score = max(best_score, record["highest_objective"])
            seed += 1
            _print_run_progress(len(records), total_runs, record, best_score)

    for label, (initial, final) in states.items():
        plot_deck(initial, image_dir / f"{label}_initial.png", f"{label} - Initial State")
        plot_deck(final, image_dir / f"{label}_final.png", f"{label} - Final State")

    plot_paths = plot_experiment_results(records, out_dir / "plots")
    results = {
        "problem": {"ship_id": ship.id, "width": ship.w, "length": ship.l,
                    "max_capacity": ship.max_capacity, "vehicle_count": len(vehicles)},
        "configuration": {
            "hc_runs_per_variant": 3, "sa_runs": 3,
            "ga_population_sizes": args.population_sizes,
            "ga_generation_counts": args.generation_counts,
            "ga_runs": len(args.population_sizes) * len(args.generation_counts),
        },
        "summary": {
            "total_runs": len(records),
            "highest_objective": max((record["highest_objective"] for record in records), default=None),
            "best_run": {
                "algorithm": max(records, key=lambda record: record["highest_objective"])["algorithm"],
                "run": max(records, key=lambda record: record["highest_objective"])["run"],
                "highest_objective": max(record["highest_objective"] for record in records),
            } if records else None,
        },
        "runs": records,
        "artifacts": {"state_images": str(image_dir), "plots": [str(p) for p in plot_paths]},
    }
    (out_dir / "experiment_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = ["algorithm", "run", "seed", "initial_objective", "final_objective", "highest_objective",
               "iterations", "duration_seconds", "final_valid", "population_size", "generations"]
    with (out_dir / "experiment_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return results


def _slug(value: str) -> str:
    return "_".join(value.lower().replace("-", " ").split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Otomatisasi eksperimen HC, SA, dan 18 konfigurasi GA.")
    parser.add_argument("--data", help="Dataset JSON (default: dataset bawaan).")
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--max-sideways-moves", type=int, default=20)
    parser.add_argument("--max-neighbor-attempts", type=int, default=20)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--initial-temperature", type=float, default=1000.0)
    parser.add_argument("--cooling-rate", type=float, default=0.99)
    parser.add_argument("--min-temperature", type=float, default=1e-3)
    parser.add_argument("--population-sizes", type=int, nargs="+", default=[20, 40, 60, 80, 100, 120])
    parser.add_argument("--generation-counts", type=int, nargs="+", default=[50, 100, 150])
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--output-dir", default="output/experiments")
    args = parser.parse_args()
    if args.max_iterations < 0 or args.max_restarts < 1:
        parser.error("max-iterations harus >= 0 dan max-restarts harus >= 1")
    if any(size < 2 for size in args.population_sizes) or any(count < 1 for count in args.generation_counts):
        parser.error("ukuran populasi minimal 2 dan jumlah generasi minimal 1")
    if len(args.population_sizes) * len(args.generation_counts) != 18:
        parser.error("Eksperimen GA harus memiliki tepat 18 kombinasi populasi dan generasi")
    if not 0 <= args.mutation_rate <= 1 or not 0 <= args.crossover_rate <= 1:
        parser.error("mutation-rate dan crossover-rate harus berada pada rentang 0 sampai 1")
    run_experiments(args)


if __name__ == "__main__":
    main()
