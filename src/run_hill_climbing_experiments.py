"""Run three reproducible experiments for each Hill-Climbing variant.

Run from the repository root:
    python -m src.run_hill_climbing_experiments

The runner writes structured JSON and CSV artifacts. Plotting is intentionally
kept separate so the report visualizations can be designed after the data has
been validated.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.dataset import get_default_problem
from src.models import CargoState
from src.hill_climbing_sideways import hill_climbing_with_sideways_move
from src.random_restart_hill_climbing import random_restart_hill_climbing
from src.stochastic_hill_climbing import stochastic_hill_climbing
from src.steepest_ascent_hill_climbing import steepest_ascent_hill_climbing

VARIANT_ORDER = (
    "steepest_ascent",
    "sideways_move",
    "stochastic",
    "random_restart",
)


def _result_record(
    variant: str,
    run_number: int,
    replicate_seed: int,
    algorithm_seed: int,
    config: dict[str, Any],
    result: Any,
    ship: Any,
) -> dict[str, Any]:
    initial_state = result.initial_state
    final_state = result.final_state
    record = {
        "variant": variant,
        "run_number": run_number,
        "replicate_seed": replicate_seed,
        "algorithm_seed": algorithm_seed,
        "configuration": config,
        "initial_state": initial_state.to_dict(),
        "final_state": final_state.to_dict(),
        "initial_objective": float(initial_state.evaluate()),
        "final_objective": float(final_state.evaluate()),
        "objective_history": [float(value) for value in result.objective_history],
        "iterations": int(result.iterations),
        "accepted_moves": int(result.accepted_moves),
        "duration_seconds": float(result.duration_seconds),
        "termination_reason": result.termination_reason,
        "metadata": result.metadata.copy(),
        "initial_valid": bool(initial_state.is_valid()),
        "final_valid": bool(final_state.is_valid()),
        "initial_total_weight": int(initial_state.get_total_weight()),
        "final_total_weight": int(final_state.get_total_weight()),
        "ship_max_capacity": int(ship.max_capacity),
        "initial_shipping_fee": int(initial_state.get_total_shipping_fee()),
        "final_shipping_fee": int(final_state.get_total_shipping_fee()),
    }
    if variant == "random_restart":
        record["objective_history_iterations"] = list(
            result.metadata["objective_history_iterations"]
        )
    else:
        record["objective_history_iterations"] = list(
            range(len(result.objective_history))
        )
    _validate_record(record)
    return record


def _validate_record(record: dict[str, Any]) -> None:
    """Validate records before they are exported for reporting/plotting."""
    if not record["initial_valid"] or not record["final_valid"]:
        raise AssertionError(
            f"State tidak valid pada {record['variant']} run {record['run_number']}"
        )

    history = record["objective_history"]
    if not history:
        raise AssertionError("Objective history tidak boleh kosong.")
    if any(later + 1e-9 < earlier for earlier, later in zip(history, history[1:])):
        raise AssertionError(
            f"Objective history menurun pada {record['variant']} "
            f"run {record['run_number']}"
        )
    if len(history) != len(record["objective_history_iterations"]):
        raise AssertionError("Panjang history dan sumbu iterasi tidak sama.")
    if record["objective_history_iterations"] != sorted(
        record["objective_history_iterations"]
    ):
        raise AssertionError("Sumbu iterasi harus monoton non-menurun.")

    if record["variant"] == "random_restart":
        expected_points = record["iterations"] + record["metadata"]["restarts_completed"]
        if len(history) != expected_points:
            raise AssertionError(
                "Panjang riwayat Random Restart tidak cocok dengan jumlah "
                "iterasi dan restart."
            )
        if record["objective_history_iterations"][-1] != record["iterations"]:
            raise AssertionError("Sumbu iterasi Random Restart tidak berakhir di total iterasi.")
        best_final = max(record["metadata"]["best_objective_per_restart"])
        if abs(record["final_objective"] - best_final) > 1e-9:
            raise AssertionError("Final objective tidak sama dengan hasil restart terbaik.")
    elif len(history) != record["iterations"] + 1:
        raise AssertionError(
            f"Panjang history {record['variant']} harus iterations + 1."
        )

    if abs(history[0] - record["initial_objective"]) > 1e-9:
        # Random Restart's aggregate history starts at its first restart's
        # initial value, so this equality is expected for this variant too.
        raise AssertionError("Titik pertama history tidak sesuai initial objective.")
    if abs(history[-1] - record["final_objective"]) > 1e-9:
        raise AssertionError("Titik terakhir history tidak sesuai final objective.")


def _write_csv_summary(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "variant", "run_number", "replicate_seed", "algorithm_seed",
        "initial_objective", "final_objective", "iterations", "accepted_moves",
        "duration_seconds", "termination_reason", "initial_valid", "final_valid",
        "initial_total_weight", "final_total_weight", "ship_max_capacity",
        "initial_shipping_fee", "final_shipping_fee", "objective_history_points",
        "metadata_json", "configuration_json",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in columns if not key.endswith("_json")}
            row["objective_history_points"] = len(record["objective_history"])
            row["metadata_json"] = json.dumps(record["metadata"], ensure_ascii=False)
            row["configuration_json"] = json.dumps(record["configuration"], ensure_ascii=False)
            writer.writerow(row)


def _write_csv_history(path: Path, records: list[dict[str, Any]]) -> None:
    columns = ["variant", "run_number", "replicate_seed", "algorithm_seed", "history_index", "iteration", "objective"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            for index, (iteration, objective) in enumerate(zip(
                record["objective_history_iterations"], record["objective_history"]
            )):
                writer.writerow({
                    "variant": record["variant"],
                    "run_number": record["run_number"],
                    "replicate_seed": record["replicate_seed"],
                    "algorithm_seed": record["algorithm_seed"],
                    "history_index": index,
                    "iteration": iteration,
                    "objective": objective,
                })


def run_experiments(
    repeats: int = 3,
    seed_base: int = 2301,
    max_iterations: int = 100,
    max_sideways_moves: int = 20,
    max_neighbor_attempts: int = 20,
    max_restarts: int = 3,
    output_dir: str | Path = "output/runs",
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats harus minimal 1.")
    if max_iterations < 0:
        raise ValueError("max_iterations tidak boleh negatif.")
    if max_sideways_moves < 0:
        raise ValueError("max_sideways_moves tidak boleh negatif.")
    if max_neighbor_attempts < 1:
        raise ValueError("max_neighbor_attempts harus minimal 1.")
    if max_restarts < 1:
        raise ValueError("max_restarts harus minimal 1.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    configuration = {
        "repeats_per_variant": repeats,
        "seed_base": seed_base,
        "max_iterations": max_iterations,
        "max_sideways_moves": max_sideways_moves,
        "max_neighbor_attempts": max_neighbor_attempts,
        "max_restarts": max_restarts,
        "max_iterations_per_restart": max_iterations,
        "use_bonus_objective": False,
        "objective": "total shipping fee of valid placed vehicles",
        "seed_policy": (
            "Each replicate uses one shared seeded initial state for Steepest "
            "Ascent, Sideways Move, and Stochastic HC. Algorithm randomness is "
            "seeded separately per variant. Random Restart generates its own "
            "random initial states and records the first restart as initial_state."
        ),
        "variant_order": list(VARIANT_ORDER),
    }

    ship, vehicles = get_default_problem()
    records: list[dict[str, Any]] = []

    algorithm_indices = {name: idx for idx, name in enumerate(VARIANT_ORDER, start=1)}
    for run_number in range(1, repeats + 1):
        replicate_seed = seed_base + run_number - 1
        random.seed(replicate_seed)
        common_initial_state = CargoState.generate_random_initial_state(
            ship=ship,
            vehicles=vehicles,
            use_bonus_objective=False,
        )
        if not common_initial_state.is_valid():
            raise RuntimeError(f"Initial state replicate {run_number} tidak valid.")

        for variant in VARIANT_ORDER:
            algorithm_seed = replicate_seed + algorithm_indices[variant] * 1000
            random.seed(algorithm_seed)
            if variant == "steepest_ascent":
                algo_config = {"max_iterations": max_iterations}
                result = steepest_ascent_hill_climbing(
                    common_initial_state.copy(), max_iterations=max_iterations
                )
            elif variant == "sideways_move":
                algo_config = {
                    "max_iterations": max_iterations,
                    "max_sideways_moves": max_sideways_moves,
                }
                result = hill_climbing_with_sideways_move(
                    common_initial_state.copy(),
                    max_iterations=max_iterations,
                    max_sideways_moves=max_sideways_moves,
                )
            elif variant == "stochastic":
                algo_config = {
                    "max_iterations": max_iterations,
                    "max_neighbor_attempts": max_neighbor_attempts,
                }
                result = stochastic_hill_climbing(
                    common_initial_state.copy(),
                    max_iterations=max_iterations,
                    max_neighbor_attempts=max_neighbor_attempts,
                )
            else:
                algo_config = {
                    "max_restarts": max_restarts,
                    "max_iterations_per_restart": max_iterations,
                    "use_bonus_objective": False,
                }
                result = random_restart_hill_climbing(
                    ship=ship,
                    vehicles=vehicles,
                    max_restarts=max_restarts,
                    max_iterations_per_restart=max_iterations,
                    use_bonus_objective=False,
                )

            record = _result_record(
                variant=variant,
                run_number=run_number,
                replicate_seed=replicate_seed,
                algorithm_seed=algorithm_seed,
                config=algo_config,
                result=result,
                ship=ship,
            )
            record["vehicle_count"] = len(vehicles)
            record["ship_dimensions"] = {"width": ship.w, "length": ship.l}
            records.append(record)
            print(
                f"[{run_number}/{repeats}] {variant:<16} "
                f"initial={record['initial_objective']:.2f} "
                f"final={record['final_objective']:.2f} "
                f"iterations={record['iterations']} "
                f"time={record['duration_seconds']:.3f}s "
                f"stop={record['termination_reason']}"
            )

    results_document = {
        "experiment": "Hill-Climbing Variants",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "problem": {
            "ship_id": ship.id,
            "ship_width": ship.w,
            "ship_length": ship.l,
            "max_capacity": ship.max_capacity,
            "vehicle_count": len(vehicles),
            "vehicle_ids": list(vehicles.keys()),
        },
        "configuration": configuration,
        "summary": {
            "variants": len(VARIANT_ORDER),
            "repeats_per_variant": repeats,
            "expected_runs": len(VARIANT_ORDER) * repeats,
            "actual_runs": len(records),
            "all_initial_states_valid": all(r["initial_valid"] for r in records),
            "all_final_states_valid": all(r["final_valid"] for r in records),
            "all_objective_histories_nondecreasing": all(
                all(b + 1e-9 >= a for a, b in zip(r["objective_history"], r["objective_history"][1:]))
                for r in records
            ),
        },
        "runs": records,
    }

    json_path = output_path / "hill_climbing_all_variants_results.json"
    summary_path = output_path / "hill_climbing_all_variants_summary.csv"
    history_path = output_path / "hill_climbing_all_variants_objective_history.csv"
    manifest_path = output_path / "hill_climbing_all_variants_manifest.json"

    json_path.write_text(
        json.dumps(results_document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv_summary(summary_path, records)
    _write_csv_history(history_path, records)
    manifest_path.write_text(
        json.dumps({
            "experiment": results_document["experiment"],
            "created_at_utc": results_document["created_at_utc"],
            "problem": results_document["problem"],
            "configuration": configuration,
            "artifact_files": [json_path.name, summary_path.name, history_path.name],
            "validation": results_document["summary"],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nArtefak eksperimen tersimpan:")
    print(f"  JSON lengkap : {json_path}")
    print(f"  Ringkasan CSV: {summary_path}")
    print(f"  Riwayat CSV  : {history_path}")
    print(f"  Manifest     : {manifest_path}")

    if len(records) != repeats * len(VARIANT_ORDER):
        raise AssertionError("Jumlah run aktual tidak sesuai konfigurasi.")
    if not results_document["summary"]["all_initial_states_valid"]:
        raise AssertionError("Ada initial state yang tidak valid.")
    if not results_document["summary"]["all_final_states_valid"]:
        raise AssertionError("Ada final state yang tidak valid.")
    if not results_document["summary"]["all_objective_histories_nondecreasing"]:
        raise AssertionError("Ada objective history yang menurun.")

    return results_document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=2301)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--max-sideways-moves", type=int, default=20)
    parser.add_argument("--max-neighbor-attempts", type=int, default=20)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--output-dir", default="output/runs")
    args = parser.parse_args()
    run_experiments(
        repeats=args.repeats,
        seed_base=args.seed_base,
        max_iterations=args.max_iterations,
        max_sideways_moves=args.max_sideways_moves,
        max_neighbor_attempts=args.max_neighbor_attempts,
        max_restarts=args.max_restarts,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()