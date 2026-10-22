import copy
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass(frozen=True)
class Vehicle:
    id: str
    w: int
    l: int
    shipping_fee: int
    weight: int
    eta: int

    def get_effective_dimensions(self, orientation: str) -> Tuple[int, int]:
        if orientation == "Horizontal":
            return (self.w, self.l)
        elif orientation == "Vertical":
            return (self.l, self.w)
        raise ValueError(f"Orientasi tidak valid: {orientation}")


@dataclass(frozen=True)
class Ship:
    w: int
    l: int
    max_capacity: int
    id: str = "Ship-1"


@dataclass
class VehiclePlacement:
    vehicle_id: str
    in_ship: bool
    x: int = -1
    y: int = -1
    orientation: str = "Horizontal"
    ship_id: Optional[str] = None

    def copy(self) -> 'VehiclePlacement':
        return VehiclePlacement(self.vehicle_id, self.in_ship, self.x, self.y, self.orientation, self.ship_id)


class CargoState:
    def __init__(self, ship: Ship | List[Ship], vehicles: Dict[str, Vehicle], placements: Dict[str, VehiclePlacement], use_bonus_objective: bool = False):
        self.ships = list(ship) if isinstance(ship, (list, tuple)) else [ship]
        if not self.ships:
            raise ValueError("Armada harus memiliki minimal satu kapal.")
        if len({item.id for item in self.ships}) != len(self.ships):
            raise ValueError("ID setiap kapal harus unik.")
        self.ship = self.ships[0]
        self._ships_by_id = {item.id: item for item in self.ships}
        self.vehicles = vehicles
        self.placements = placements
        self.use_bonus_objective = use_bonus_objective
        self._cached_value: Optional[float] = None

    def copy(self) -> 'CargoState':
        new_placements = {vid: p.copy() for vid, p in self.placements.items()}
        return CargoState(self.ships, self.vehicles, new_placements, self.use_bonus_objective)

    def _resolve_ship(self, ship_id: Optional[str]) -> Ship:
        if ship_id is None:
            return self.ship
        try:
            return self._ships_by_id[ship_id]
        except KeyError as exc:
            raise ValueError(f"Kapal tidak dikenal: {ship_id}") from exc

    def invalidate_cache(self):
        self._cached_value = None

    def get_grid(self, ship_id: Optional[str] = None) -> List[List[Optional[str]]]:
        ship = self._resolve_ship(ship_id)
        grid = [[None for _ in range(ship.w)] for _ in range(ship.l)]
        for vid, p in self.placements.items():
            if p.in_ship and (p.ship_id or self.ship.id) == ship.id:
                v = self.vehicles[vid]
                eff_w, eff_l = v.get_effective_dimensions(p.orientation)
                for dy in range(eff_l):
                    for dx in range(eff_w):
                        gx, gy = p.x + dx, p.y + dy
                        if 0 <= gx < ship.w and 0 <= gy < ship.l:
                            grid[gy][gx] = vid
        return grid

    def get_total_weight(self, ship_id: Optional[str] = None) -> int:
        return sum(self.vehicles[vid].weight for vid, p in self.placements.items()
                   if p.in_ship and (ship_id is None or (p.ship_id or self.ship.id) == ship_id))

    def get_total_shipping_fee(self) -> int:
        return sum(self.vehicles[vid].shipping_fee for vid, p in self.placements.items() if p.in_ship)

    def count_violations(self) -> Tuple[int, int, int]:
        grid_counts = {
            ship.id: [[0 for _ in range(ship.w)] for _ in range(ship.l)]
            for ship in self.ships
        }
        ship_weights = {ship.id: 0 for ship in self.ships}
        out_of_bounds, overlap = 0, 0

        for vid, p in self.placements.items():
            if not p.in_ship: continue
            v = self.vehicles[vid]
            ship_id = p.ship_id or self.ship.id
            ship = self._resolve_ship(ship_id)
            grid_count = grid_counts[ship.id]
            ship_weights[ship.id] += v.weight
            eff_w, eff_l = v.get_effective_dimensions(p.orientation)

            if p.x < 0 or p.y < 0 or (p.x + eff_w) > ship.w or (p.y + eff_l) > ship.l:
                for dy in range(eff_l):
                    for dx in range(eff_w):
                        gx, gy = p.x + dx, p.y + dy
                        if gx < 0 or gx >= ship.w or gy < 0 or gy >= ship.l:
                            out_of_bounds += 1
                        else:
                            grid_count[gy][gx] += 1
                            if grid_count[gy][gx] > 1: overlap += 1
            else:
                for dy in range(eff_l):
                    for dx in range(eff_w):
                        gx, gy = p.x + dx, p.y + dy
                        grid_count[gy][gx] += 1
                        if grid_count[gy][gx] > 1: overlap += 1

        excess_weight = sum(
            max(0, ship_weights[ship.id] - ship.max_capacity)
            for ship in self.ships
        )
        return out_of_bounds, overlap, excess_weight

    def is_valid(self) -> bool:
        oob, ovl, ex_w = self.count_violations()
        return (oob == 0) and (ovl == 0) and (ex_w == 0)

    def can_place_vehicle(self, vid: str, x: int, y: int, orientation: str,
                          ignore_vids: Optional[set] = None,
                          ship_id: Optional[str] = None) -> bool:
        ignore_vids = {vid} if ignore_vids is None else set(ignore_vids) | {vid}
        v = self.vehicles[vid]
        current = self.placements.get(vid)
        target_ship = self._resolve_ship(ship_id or (current.ship_id if current and current.in_ship else None))
        eff_w, eff_l = v.get_effective_dimensions(orientation)

        if x < 0 or y < 0 or (x + eff_w) > target_ship.w or (y + eff_l) > target_ship.l: return False

        current_weight = sum(
            self.vehicles[other_id].weight for other_id, p in self.placements.items()
            if p.in_ship and other_id not in ignore_vids
            and (p.ship_id or self.ship.id) == target_ship.id
        )
        if current_weight + v.weight > target_ship.max_capacity: return False

        for other_id, p in self.placements.items():
            if (not p.in_ship or other_id in ignore_vids
                    or (p.ship_id or self.ship.id) != target_ship.id): continue
            other_v = self.vehicles[other_id]
            ow, ol = other_v.get_effective_dimensions(p.orientation)
            no_overlap = (x + eff_w <= p.x) or (p.x + ow <= x) or (y + eff_l <= p.y) or (p.y + ol <= y)
            if not no_overlap: return False
        return True

    def evaluate(self) -> float:
        """
        REVISI: Memberikan penalti ekstrem untuk state tidak valid agar tidak pernah terpilih.
        """
        if self._cached_value is not None:
            return self._cached_value

        oob, ovl, ex_w = self.count_violations()
        
        # Penalti Ekstrem: Mencegah state rusak (terutama hasil GA) mengalahkan state valid
        if oob > 0 or ovl > 0 or ex_w > 0:
            self._cached_value = -1e9  # Sangat rendah, dipastikan kalah
            return self._cached_value

        total_fee = float(self.get_total_shipping_fee())
        
        if not self.use_bonus_objective:
            self._cached_value = total_fee
        else:
            bonus_score = 0.0
            for vid, p in self.placements.items():
                if p.in_ship:
                    v = self.vehicles[vid]
                    urgency_bonus = v.shipping_fee * (1.0 / max(1, v.eta)) * 0.1
                    bonus_score += v.shipping_fee + urgency_bonus
            self._cached_value = round(bonus_score, 4)

        return self._cached_value

    @classmethod
    def generate_random_initial_state(cls, ship: Ship | List[Ship], vehicles: Dict[str, Vehicle], use_bonus_objective: bool = False) -> 'CargoState':
        ships = list(ship) if isinstance(ship, (list, tuple)) else [ship]
        if len(ships) > 1:
            placements = {vid: VehiclePlacement(vid, False) for vid in vehicles}
            state = cls(ships, vehicles, placements, use_bonus_objective)
            vehicle_ids = list(vehicles)
            random.shuffle(vehicle_ids)
            for vid in vehicle_ids:
                if random.random() >= 0.85:
                    continue
                options = []
                for target in ships:
                    for orientation in ("Horizontal", "Vertical"):
                        width, length = vehicles[vid].get_effective_dimensions(orientation)
                        if width <= target.w and length <= target.l:
                            options.append((target, orientation))
                random.shuffle(options)
                for target, orientation in options:
                    width, length = vehicles[vid].get_effective_dimensions(orientation)
                    coords = [(x, y) for x in range(target.w - width + 1)
                              for y in range(target.l - length + 1)]
                    random.shuffle(coords)
                    for x, y in coords[:15]:
                        if state.can_place_vehicle(vid, x, y, orientation, ship_id=target.id):
                            state.placements[vid] = VehiclePlacement(
                                vid, True, x, y, orientation, target.id
                            )
                            break
                    if state.placements[vid].in_ship:
                        break
            return state
        placements = {vid: VehiclePlacement(vid, False, -1, -1, "Horizontal") for vid in vehicles}
        state = cls(ships[0], vehicles, placements, use_bonus_objective)
        vehicle_ids = list(vehicles.keys())
        random.shuffle(vehicle_ids)

        for vid in vehicle_ids:
            if random.random() < 0.85:
                orientation = random.choice(["Horizontal", "Vertical"])
                v = vehicles[vid]
                eff_w, eff_l = v.get_effective_dimensions(orientation)
                if eff_w <= ship.w and eff_l <= ship.l:
                    candidate_coords = [(x, y) for x in range(ship.w - eff_w + 1) for y in range(ship.l - eff_l + 1)]
                    random.shuffle(candidate_coords)
                    for cx, cy in candidate_coords[:15]:
                        if state.can_place_vehicle(vid, cx, cy, orientation):
                            state.placements[vid] = VehiclePlacement(vid, True, cx, cy, orientation)
                            break
        state.invalidate_cache()
        return state

    def get_random_neighbor(self, max_attempts: int = 50) -> Optional['CargoState']:
        """
        REVISI: Mengembalikan None jika gagal menemukan state baru, BUKAN self.copy().
        Ini mencegah SA mendeteksi Delta E = 0 secara artifisial.
        """
        if len(self.ships) > 1:
            return self._get_random_fleet_neighbor(max_attempts)
        vids = list(self.vehicles.keys())

        for _ in range(max_attempts):
            move_type = random.choice([1, 2, 3])
            neighbor = self.copy()

            if move_type == 1 and len(vids) >= 2:
                v1_id, v2_id = random.sample(vids, 2)
                p1, p2 = neighbor.placements[v1_id], neighbor.placements[v2_id]
                
                # In-In Swap
                if p1.in_ship and p2.in_ship:
                    if neighbor.can_place_vehicle(v1_id, p2.x, p2.y, p1.orientation, ignore_vids={v1_id, v2_id}) and \
                       neighbor.can_place_vehicle(v2_id, p1.x, p1.y, p2.orientation, ignore_vids={v1_id, v2_id}):
                        p1.x, p2.x = p2.x, p1.x
                        p1.y, p2.y = p2.y, p1.y
                        neighbor.invalidate_cache()
                        return neighbor
                # In-Out Swap
                elif p1.in_ship != p2.in_ship:
                    in_id, out_id = (v1_id, v2_id) if p1.in_ship else (v2_id, v1_id)
                    p_in, p_out = neighbor.placements[in_id], neighbor.placements[out_id]
                    for orient in [p_out.orientation, "Vertical" if p_out.orientation == "Horizontal" else "Horizontal"]:
                        if neighbor.can_place_vehicle(out_id, p_in.x, p_in.y, orient, ignore_vids={in_id, out_id}):
                            p_out.in_ship, p_out.x, p_out.y, p_out.orientation = True, p_in.x, p_in.y, orient
                            p_in.in_ship, p_in.x, p_in.y = False, -1, -1
                            neighbor.invalidate_cache()
                            return neighbor

            elif move_type == 2:
                vid = random.choice(vids)
                p = neighbor.placements[vid]
                v = self.vehicles[vid]
                eff_w, eff_l = v.get_effective_dimensions(p.orientation)

                if not p.in_ship:
                    if eff_w <= self.ship.w and eff_l <= self.ship.l:
                        valid_coords = [(x, y) for x in range(self.ship.w - eff_w + 1) for y in range(self.ship.l - eff_l + 1) if neighbor.can_place_vehicle(vid, x, y, p.orientation)]
                        if valid_coords:
                            p.in_ship, (p.x, p.y) = True, random.choice(valid_coords)
                            neighbor.invalidate_cache()
                            return neighbor
                else:
                    if random.random() < 0.25: # Coba keluarkan
                        p.in_ship, p.x, p.y = False, -1, -1
                        neighbor.invalidate_cache()
                        return neighbor
                    else: # Pindah koordinat
                        valid_coords = [(x, y) for x in range(self.ship.w - eff_w + 1) for y in range(self.ship.l - eff_l + 1) if (x != p.x or y != p.y) and neighbor.can_place_vehicle(vid, x, y, p.orientation)]
                        if valid_coords:
                            p.x, p.y = random.choice(valid_coords)
                            neighbor.invalidate_cache()
                            return neighbor

            elif move_type == 3:
                vid = random.choice(vids)
                p = neighbor.placements[vid]
                new_orient = "Vertical" if p.orientation == "Horizontal" else "Horizontal"
                
                if p.in_ship and neighbor.can_place_vehicle(vid, p.x, p.y, new_orient):
                    p.orientation = new_orient
                    neighbor.invalidate_cache()
                    return neighbor
                elif not p.in_ship: # Ubah orientasi yang ada di luar
                    p.orientation = new_orient
                    neighbor.invalidate_cache()
                    return neighbor

        return None  # Mengembalikan None jika mentok

    def get_all_neighbors(self) -> List['CargoState']:
        """
        REVISI: Menghapus sampling. Mengecek SELURUH koordinat, orientasi, dan Swap secara exhaustive.
        Sangat krusial untuk Steepest Ascent Hill-Climbing agar tidak bias.
        """
        if len(self.ships) > 1:
            return self._get_all_fleet_neighbors()
        neighbors: List[CargoState] = []
        vids = list(self.vehicles.keys())

        # 1. Move 3: Ubah Orientasi
        for vid in vids:
            p = self.placements[vid]
            new_orient = "Vertical" if p.orientation == "Horizontal" else "Horizontal"
            
            nb = self.copy()
            if p.in_ship and self.can_place_vehicle(vid, p.x, p.y, new_orient):
                nb.placements[vid].orientation = new_orient
                nb.invalidate_cache()
                neighbors.append(nb)
            elif not p.in_ship:
                nb.placements[vid].orientation = new_orient
                nb.invalidate_cache()
                neighbors.append(nb)

        # 2. Move 2: Pindah Koordinat / Masuk / Keluar (Exhaustive, tanpa max_coord_samples)
        for vid in vids:
            p = self.placements[vid]
            v = self.vehicles[vid]
            for orient in ["Horizontal", "Vertical"]:
                eff_w, eff_l = v.get_effective_dimensions(orient)
                if eff_w > self.ship.w or eff_l > self.ship.l: continue
                
                # Cek SEMUA koordinat yang mungkin
                for cx in range(self.ship.w - eff_w + 1):
                    for cy in range(self.ship.l - eff_l + 1):
                        if p.in_ship and p.x == cx and p.y == cy and p.orientation == orient:
                            continue
                        if self.can_place_vehicle(vid, cx, cy, orient):
                            nb = self.copy()
                            nb.placements[vid].in_ship, nb.placements[vid].x, nb.placements[vid].y, nb.placements[vid].orientation = True, cx, cy, orient
                            nb.invalidate_cache()
                            neighbors.append(nb)
                            
            if p.in_ship: # Opsi Keluarkan
                nb = self.copy()
                nb.placements[vid].in_ship, nb.placements[vid].x, nb.placements[vid].y = False, -1, -1
                nb.invalidate_cache()
                neighbors.append(nb)

        # 3. Move 1: Swap (In-Out dan In-In)
        in_vids = [vid for vid in vids if self.placements[vid].in_ship]
        out_vids = [vid for vid in vids if not self.placements[vid].in_ship]

        # In-Out Swap
        for in_id in in_vids:
            p_in = self.placements[in_id]
            for out_id in out_vids:
                for orient in ["Horizontal", "Vertical"]:
                    if self.can_place_vehicle(out_id, p_in.x, p_in.y, orient, ignore_vids={in_id, out_id}):
                        nb = self.copy()
                        nb.placements[in_id].in_ship, nb.placements[in_id].x, nb.placements[in_id].y = False, -1, -1
                        nb.placements[out_id].in_ship, nb.placements[out_id].x, nb.placements[out_id].y, nb.placements[out_id].orientation = True, p_in.x, p_in.y, orient
                        nb.invalidate_cache()
                        neighbors.append(nb)
        
        # REVISI: In-In Swap (Menukar 2 kendaraan yang ada di dalam kapal)
        for i in range(len(in_vids)):
            for j in range(i + 1, len(in_vids)):
                v1_id, v2_id = in_vids[i], in_vids[j]
                p1, p2 = self.placements[v1_id], self.placements[v2_id]
                if self.can_place_vehicle(v1_id, p2.x, p2.y, p1.orientation, ignore_vids={v1_id, v2_id}) and \
                   self.can_place_vehicle(v2_id, p1.x, p1.y, p2.orientation, ignore_vids={v1_id, v2_id}):
                    nb = self.copy()
                    nb.placements[v1_id].x, nb.placements[v2_id].x = p2.x, p1.x
                    nb.placements[v1_id].y, nb.placements[v2_id].y = p2.y, p1.y
                    nb.invalidate_cache()
                    neighbors.append(nb)

        return neighbors

    def _get_all_fleet_neighbors(self) -> List['CargoState']:
        neighbors = []
        for vid, placement in self.placements.items():
            if placement.in_ship:
                removed = self.copy()
                removed.placements[vid] = VehiclePlacement(vid, False)
                removed.invalidate_cache()
                neighbors.append(removed)

            vehicle = self.vehicles[vid]
            for ship in self.ships:
                for orientation in ("Horizontal", "Vertical"):
                    width, length = vehicle.get_effective_dimensions(orientation)
                    if width > ship.w or length > ship.l:
                        continue
                    for x in range(ship.w - width + 1):
                        for y in range(ship.l - length + 1):
                            if (placement.in_ship and placement.x == x and placement.y == y
                                    and placement.orientation == orientation
                                    and (placement.ship_id or self.ship.id) == ship.id):
                                continue
                            if not self.can_place_vehicle(
                                vid, x, y, orientation, ship_id=ship.id
                            ):
                                continue
                            neighbor = self.copy()
                            neighbor.placements[vid] = VehiclePlacement(
                                vid, True, x, y, orientation, ship.id
                            )
                            neighbor.invalidate_cache()
                            neighbors.append(neighbor)
        return neighbors

    def _get_random_fleet_neighbor(self, max_attempts: int) -> Optional['CargoState']:
        vehicle_ids = list(self.vehicles)
        for _ in range(max_attempts):
            vid = random.choice(vehicle_ids)
            placement = self.placements[vid]
            neighbor = self.copy()
            if placement.in_ship and random.random() < 0.2:
                neighbor.placements[vid] = VehiclePlacement(vid, False)
                neighbor.invalidate_cache()
                return neighbor

            vehicle = self.vehicles[vid]
            target = random.choice(self.ships)
            orientation = random.choice(("Horizontal", "Vertical"))
            width, length = vehicle.get_effective_dimensions(orientation)
            if width > target.w or length > target.l:
                continue
            x = random.randrange(target.w - width + 1)
            y = random.randrange(target.l - length + 1)
            if (placement.in_ship and placement.x == x and placement.y == y
                    and placement.orientation == orientation
                    and (placement.ship_id or self.ship.id) == target.id):
                continue
            if neighbor.can_place_vehicle(vid, x, y, orientation, ship_id=target.id):
                neighbor.placements[vid] = VehiclePlacement(
                    vid, True, x, y, orientation, target.id
                )
                neighbor.invalidate_cache()
                return neighbor
        return None

    def to_dict(self) -> Dict:
        data = {
            "ship": {
                "id": self.ship.id,
                "w": self.ship.w,
                "l": self.ship.l,
                "max_capacity": self.ship.max_capacity,
            },
            "vehicles": {
                vid: {
                    "id": vehicle.id,
                    "w": vehicle.w,
                    "l": vehicle.l,
                    "shipping_fee": vehicle.shipping_fee,
                    "weight": vehicle.weight,
                    "eta": vehicle.eta,
                }
                for vid, vehicle in self.vehicles.items()
            },
            "placements": {
                vid: {
                    "vehicle_id": placement.vehicle_id,
                    "in_ship": placement.in_ship,
                    "x": placement.x,
                    "y": placement.y,
                    "orientation": placement.orientation,
                    **({"ship_id": placement.ship_id or self.ship.id} if len(self.ships) > 1 and placement.in_ship else {}),
                }
                for vid, placement in self.placements.items()
            },
            "use_bonus_objective": self.use_bonus_objective,
            "is_valid": self.is_valid(),
            "objective": self.evaluate(),
            "total_weight": self.get_total_weight(),
            "total_shipping_fee": self.get_total_shipping_fee(),
            "ship_totals": {
                ship.id: {
                    "total_weight": self.get_total_weight(ship.id),
                    "shipping_fee": sum(
                        self.vehicles[vid].shipping_fee
                        for vid, placement in self.placements.items()
                        if placement.in_ship and (placement.ship_id or self.ship.id) == ship.id
                    ),
                }
                for ship in self.ships
            },
        }
        if len(self.ships) > 1:
            data["ships"] = [
                {"id": ship.id, "w": ship.w, "l": ship.l,
                 "max_capacity": ship.max_capacity}
                for ship in self.ships
            ]
        return data
