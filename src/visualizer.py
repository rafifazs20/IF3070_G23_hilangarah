from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.models import CargoState, Ship, Vehicle, VehiclePlacement


def plot_deck(state: CargoState, output_path: str | Path, title: str = "",
              ship_id: str | None = None) -> Path:
    ships = [state._resolve_ship(ship_id)] if ship_id else state.ships
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(ships), squeeze=False,
        figsize=(max(7, ships[0].w * 0.65) * len(ships), max(6, ships[0].l * 0.42)),
    )
    colors = plt.get_cmap("tab20")
    for ship, ax in zip(ships, axes[0]):
        onboard = [
            (index, vehicle_id, placement)
            for index, (vehicle_id, placement) in enumerate(state.placements.items())
            if placement.in_ship and (placement.ship_id or state.ship.id) == ship.id
        ]
        for index, vehicle_id, placement in onboard:
            vehicle = state.vehicles[vehicle_id]
            width, length = vehicle.get_effective_dimensions(placement.orientation)
            ax.add_patch(Rectangle((placement.x, placement.y), width, length,
                                   facecolor=colors(index % 20), edgecolor="#253247", linewidth=1.3))
            ax.text(placement.x + width / 2, placement.y + length / 2, vehicle_id,
                    ha="center", va="center", fontsize=8, color="#111827", clip_on=True)
        ship_fee = sum(state.vehicles[vid].shipping_fee for _, vid, _ in onboard)
        ship_weight = state.get_total_weight(ship.id)
        ax.set_xlim(0, ship.w)
        ax.set_ylim(ship.l, 0)
        ax.set_xticks(range(ship.w + 1))
        ax.set_yticks(range(ship.l + 1))
        ax.grid(True, color="#9ca3af", linewidth=0.45)
        ax.set_aspect("equal")
        ax.set_xlabel("Lebar geladak (x)")
        ax.set_ylabel("Panjang geladak (y)")
        ax.set_title(f"{ship.id}\n{len(onboard)} kendaraan | fee {ship_fee} | berat {ship_weight}/{ship.max_capacity}")
    if title:
        fig.suptitle(f"{title} | objective armada {state.evaluate():.2f}")
    elif len(ships) > 1:
        fig.suptitle(f"Visualisasi Armada | objective total {state.evaluate():.2f}")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_experiment_results(records: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record.get("algorithm", record.get("variant", "Algoritma"))].append(record)

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
    ships = [Ship(
        w=item["w"], l=item["l"], max_capacity=item["max_capacity"],
        id=item.get("id", f"Ship-{index + 1}"),
    ) for index, item in enumerate(data.get("ships", [ship_data]))]
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
            ship_id=placement.get("ship_id"),
        )
        for vehicle_id, placement in data["placements"].items()
    }
    return CargoState(ships if len(ships) > 1 else ships[0], vehicles, placements,
                      data.get("use_bonus_objective", False))


def _slug(value: str) -> str:
    return "_".join(value.lower().replace("-", " ").split())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualisasikan eksperimen tugas utama dan bonus Multi-kapal."
    )
    parser.add_argument("json_file", nargs="?",
                        help="JSON tertentu; tanpa path, visualisasikan hasil single_ship dan multi_ship.")
    parser.add_argument("--output-dir", help="Folder plot untuk satu JSON yang dipilih.")
    args = parser.parse_args()
    if args.json_file:
        json_paths = [Path(args.json_file)]
    else:
        json_paths = [
            Path("output/experiments/single_ship/experiment_results.json"),
            Path("output/experiments/multi_ship/experiment_results.json"),
            Path("output/experiments/experiment_results.json"),
            Path("output/runs/hill_climbing_all_variants_results.json"),
        ]
        json_paths = [path for path in json_paths if path.exists()]
    if not json_paths:
        raise SystemExit(
            "JSON hasil eksperimen belum ditemukan. Jalankan `python -m src.run_experiments` "
            "untuk tugas utama, atau tambahkan `--num-ships 2` untuk bonus Multi-kapal."
        )

    for json_path in json_paths:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        default_folder = "visualizations" if json_path.parent == Path("output/runs") else "plots"
        output_dir = Path(args.output_dir) if args.output_dir and len(json_paths) == 1 else json_path.parent / default_folder
        plots = plot_experiment_results(data["runs"], output_dir)
        state_dir = output_dir / "states"
        state_count = 0
        for index, run in enumerate(data["runs"], start=1):
            if "initial_state" not in run or "final_state" not in run:
                continue
            algorithm_name = run.get("algorithm", run.get("variant", "algorithm"))
            name = _slug(algorithm_name)
            run_number = run.get("run", index)
            initial = state_from_dict(run["initial_state"])
            final = state_from_dict(run["final_state"])
            suffix = "fleet" if len(initial.ships) > 1 else _slug(initial.ship.id)
            plot_deck(initial, state_dir / f"{name}_run_{run_number}_{suffix}_initial.png",
                      f"{algorithm_name} run {run_number} - Initial State")
            plot_deck(final, state_dir / f"{name}_run_{run_number}_{suffix}_final.png",
                      f"{algorithm_name} run {run_number} - Final State")
            state_count += 2
        print(f"JSON: {json_path}")
        print(f"Grafik eksperimen: {len(plots)} file di {output_dir}")
        print(f"Visualisasi grid: {state_count} file di {state_dir}")


if __name__ == "__main__":
    main()
