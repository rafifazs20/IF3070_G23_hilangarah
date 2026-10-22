import json
import random
from pathlib import Path

import matplotlib.pyplot as plt

from src.dataset import get_default_problem
from src.models import CargoState
from src.steepest_ascent_hill_climbing import steepest_ascent_hill_climbing


def run_experiments():
    output_dir = Path("output")
    runs_dir = output_dir / "runs"
    plots_dir = output_dir / "plots"

    runs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    seeds = [2301, 2302, 2303]
    records = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for run_number, seed in enumerate(seeds, start=1):
        random.seed(seed)

        ship, vehicles = get_default_problem()

        initial_state = CargoState.generate_random_initial_state(
            ship,
            vehicles,
        )

        result = steepest_ascent_hill_climbing(
            initial_state=initial_state,
            max_iterations=1000,
        )

        record = result.to_dict()
        record.update({
            "run_number": run_number,
            "seed": seed,
            "initial_objective": result.initial_state.evaluate(),
            "final_objective": result.final_state.evaluate(),
            "initial_valid": result.initial_state.is_valid(),
            "final_valid": result.final_state.is_valid(),
            "final_total_weight": result.final_state.get_total_weight(),
            "max_capacity": ship.max_capacity,
            "final_shipping_fee": result.final_state.get_total_shipping_fee(),
        })

        records.append(record)

        iterations = range(len(result.objective_history))

        ax.plot(
            iterations,
            result.objective_history,
            marker="o",
            markersize=3,
            label=f"Run {run_number}",
        )

        print(f"\nRun {run_number}")
        print(f"Seed              : {seed}")
        print(f"Initial objective : {result.initial_state.evaluate()}")
        print(f"Final objective   : {result.final_state.evaluate()}")
        print(f"Final valid       : {result.final_state.is_valid()}")
        print(f"Iterations        : {result.iterations}")
        print(f"Accepted moves    : {result.accepted_moves}")
        print(f"Duration (s)      : {result.duration_seconds:.4f}")
        print(f"Termination       : {result.termination_reason}")

        if result.termination_reason != "local_optimum":
            print("WARNING: Run belum terkonfirmasi mencapai optimum lokal.")

    results_path = runs_dir / "hill_climbing_results.json"

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)

    ax.set_title("Steepest Ascent Hill-Climbing")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Function (Shipping Fee)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = plots_dir / "hill_climbing_objective.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"\nResults saved to: {results_path}")
    print(f"Plot saved to   : {plot_path}")


if __name__ == "__main__":
    run_experiments()