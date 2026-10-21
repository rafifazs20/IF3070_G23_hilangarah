import json
from typing import Dict, Tuple
from src.models import Ship, Vehicle

def get_default_problem() -> Tuple[Ship, Dict[str, Vehicle]]:
    """
    Menghasilkan studi kasus standar PT Intim Line sesuai contoh tabel variabel
    pada dokumen spesifikasi (Ukuran geladak kapal w:10, l:20, MaxCapacity: 500,
    serta daftar kendaraan dari berbagai dealer).
    """
    ship = Ship(w=10, l=20, max_capacity=500, id="IntimLine-Deck-01")

    raw_vehicles = [
        {"id": "Owen",   "w": 3, "l": 4, "fee": 60,  "weight": 50, "eta": 3},
        {"id": "V02",    "w": 3, "l": 5, "fee": 85,  "weight": 65, "eta": 2},
        {"id": "V03",    "w": 2, "l": 4, "fee": 45,  "weight": 35, "eta": 4},
        {"id": "V04",    "w": 4, "l": 6, "fee": 130, "weight": 95, "eta": 1},
        {"id": "V05",    "w": 2, "l": 3, "fee": 35,  "weight": 25, "eta": 5},
        {"id": "V06",    "w": 3, "l": 4, "fee": 65,  "weight": 45, "eta": 2},
        {"id": "V07",    "w": 4, "l": 5, "fee": 110, "weight": 80, "eta": 3},
        {"id": "V08",    "w": 2, "l": 5, "fee": 55,  "weight": 40, "eta": 2},
        {"id": "V09",    "w": 3, "l": 6, "fee": 95,  "weight": 70, "eta": 1},
        {"id": "V10",    "w": 2, "l": 3, "fee": 40,  "weight": 30, "eta": 4},
        {"id": "V11",    "w": 3, "l": 3, "fee": 50,  "weight": 40, "eta": 3},
        {"id": "V12",    "w": 4, "l": 4, "fee": 90,  "weight": 60, "eta": 2},
        {"id": "V13",    "w": 2, "l": 4, "fee": 48,  "weight": 35, "eta": 5},
        {"id": "V14",    "w": 3, "l": 5, "fee": 82,  "weight": 60, "eta": 1},
        {"id": "V15",    "w": 2, "l": 2, "fee": 28,  "weight": 20, "eta": 3},
        {"id": "V16",    "w": 3, "l": 4, "fee": 62,  "weight": 48, "eta": 2},
        {"id": "V17",    "w": 4, "l": 5, "fee": 105, "weight": 75, "eta": 4},
        {"id": "V18",    "w": 2, "l": 3, "fee": 38,  "weight": 28, "eta": 2},
    ]

    vehicles = {
        item["id"]: Vehicle(
            id=item["id"],
            w=item["w"],
            l=item["l"],
            shipping_fee=item["fee"],
            weight=item["weight"],
            eta=item["eta"]
        )
        for item in raw_vehicles
    }
    return ship, vehicles


def save_problem_to_json(filepath: str, ship: Ship, vehicles: Dict[str, Vehicle]):
    data = {
        "ship": {"id": ship.id, "Dimensions": {"w": ship.w, "l": ship.l}, "MaxCapacity": ship.max_capacity},
        "vehicles": [
            {
                "id": v.id,
                "Dimensions": {"w": v.w, "l": v.l},
                "ShippingFee": v.shipping_fee,
                "Weight": v.weight,
                "ETA": v.eta
            }
            for v in vehicles.values()
        ]
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_problem_from_json(filepath: str) -> Tuple[Ship, Dict[str, Vehicle]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    s_data = data["ship"]
    ship = Ship(
        w=s_data["Dimensions"]["w"],
        l=s_data["Dimensions"]["l"],
        max_capacity=s_data["MaxCapacity"],
        id=s_data.get("id", "Ship-1")
    )
    vehicles = {}
    for v_data in data["vehicles"]:
        vid = v_data["id"]
        vehicles[vid] = Vehicle(
            id=vid,
            w=v_data["Dimensions"]["w"],
            l=v_data["Dimensions"]["l"],
            shipping_fee=v_data["ShippingFee"],
            weight=v_data["Weight"],
            eta=v_data["ETA"]
        )
    return ship, vehicles