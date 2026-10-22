from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.models import CargoState, Ship, Vehicle, VehiclePlacement


def plot_deck(state: CargoState, output_path: str | Path, title: str = "") -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, state.ship.w * 0.65), max(6, state.ship.l * 0.42)))
    colors = plt.get_cmap("tab20")
    for index, (vehicle_id, placement) in enumerate(state.placements.items()):
        if not placement.in_ship:
            continue
        vehicle = state.vehicles[vehicle_id]
        width, length = vehicle.get_effective_dimensions(placement.orientation)
        ax.add_patch(Rectangle((placement.x, placement.y), width, length,
                               facecolor=colors(index % 20), edgecolor="#253247", linewidth=1.3))
        ax.text(placement.x + width / 2, placement.y + length / 2, vehicle_id,
                ha="center", va="center", fontsize=8, color="#111827", clip_on=True)

    ax.set_xlim(0, state.ship.w)
    ax.set_ylim(state.ship.l, 0)
    ax.set_xticks(range(state.ship.w + 1))
    ax.set_yticks(range(state.ship.l + 1))
    ax.grid(True, color="#9ca3af", linewidth=0.45)
    ax.set_aspect("equal")
    ax.set_xlabel("Lebar geladak (x)")
    ax.set_ylabel("Panjang geladak (y)")
    ax.set_title(title or f"{state.ship.id} | objektif {state.evaluate():.2f}")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_experiment_results(records: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["algorithm"]].append(record)

    outputs = []
    for algorithm, runs in grouped.items():
        fig, ax = plt.subplots(figsize=(9, 5.5))
        if algorithm == "Genetic Algorithm":
            for population in sorted({run["population_size"] for run in runs}):
                selected = sorted((run for run in runs if run["population_size"] == population),
                                  key=lambda run: run["generations"])
                ax.plot([run["generations"] for run in selected],
                        [run["final_objective"] for run in selected], marker="o",
                        label=f"Populasi {population}")
            ax.set_xlabel("Jumlah generasi")
            ax.set_ylabel("Fitness terbaik")
            ax.legend(title="Ukuran populasi")
        else:
            for run in runs:
                ax.plot(range(len(run["objective_history"])), run["objective_history"],
                        alpha=0.35, linewidth=1)
            max_length = max(len(run["objective_history"]) for run in runs)
            averages = []
            for index in range(max_length):
                values = [run["objective_history"][index] for run in runs
                          if index < len(run["objective_history"])]
                averages.append(sum(values) / len(values))
            ax.plot(range(len(averages)), averages, color="#111827", linewidth=2.5,
                    label="Rata-rata")
            ax.set_xlabel("Iterasi")
            ax.set_ylabel("Nilai objektif")
            ax.legend()
        ax.set_title(f"Riwayat eksperimen - {algorithm}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        output = out_dir / f"{_slug(algorithm)}_experiment.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
        outputs.append(output)
    return outputs


def state_from_dict(data: dict[str, Any]) -> CargoState:
    ship_data = data["ship"]
    ship = Ship(
        w=ship_data["w"], l=ship_data["l"],
        max_capacity=ship_data["max_capacity"], id=ship_data.get("id", "Ship-1"),
    )
    vehicles = {
        vehicle_id: Vehicle(
            id=vehicle["id"], w=vehicle["w"], l=vehicle["l"],
            shipping_fee=vehicle["shipping_fee"], weight=vehicle["weight"], eta=vehicle["eta"],
        )
        for vehicle_id, vehicle in data["vehicles"].items()
    }
    placements = {
        vehicle_id: VehiclePlacement(
            vehicle_id=placement["vehicle_id"], in_ship=placement["in_ship"],
            x=placement["x"], y=placement["y"], orientation=placement["orientation"],
        )
        for vehicle_id, placement in data["placements"].items()
    }
    return CargoState(ship, vehicles, placements, data.get("use_bonus_objective", False))


def _slug(value: str) -> str:
    return "_".join(value.lower().replace("-", " ").split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualisasikan hasil eksperimen dari file JSON.")
    parser.add_argument("json_file", nargs="?", default="output/experiments/experiment_results.json",
                        help="File JSON hasil eksperimen (default: output/experiments/experiment_results.json).")
    parser.add_argument("--output-dir", default="output/experiments/plots")
    args = parser.parse_args()
    json_path = Path(args.json_file)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    plots = plot_experiment_results(data["runs"], output_dir)
    state_dir = output_dir / "states"
    state_count = 0
    for index, run in enumerate(data["runs"], start=1):
        if "initial_state" not in run or "final_state" not in run:
            continue
        name = _slug(run["algorithm"])
        run_number = run.get("run", index)
        initial = state_from_dict(run["initial_state"])
        final = state_from_dict(run["final_state"])
        plot_deck(initial, state_dir / f"{name}_run_{run_number}_initial.png",
                  f"{run['algorithm']} run {run_number} - Initial State")
        plot_deck(final, state_dir / f"{name}_run_{run_number}_final.png",
                  f"{run['algorithm']} run {run_number} - Final State")
        state_count += 2
    print(f"JSON dibaca: {json_path}")
    print(f"Grafik eksperimen: {len(plots)} file di {output_dir}")
    print(f"Visualisasi grid: {state_count} file di {state_dir}")


if __name__ == "__main__":
    main()
