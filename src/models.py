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

    def copy(self) -> 'VehiclePlacement':
        return VehiclePlacement(self.vehicle_id, self.in_ship, self.x, self.y, self.orientation)


class CargoState:
    def __init__(self, ship: Ship, vehicles: Dict[str, Vehicle], placements: Dict[str, VehiclePlacement], use_bonus_objective: bool = False):
        self.ship = ship
        self.vehicles = vehicles
        self.placements = placements
        self.use_bonus_objective = use_bonus_objective
        self._cached_value: Optional[float] = None

    def copy(self) -> 'CargoState':
        new_placements = {vid: p.copy() for vid, p in self.placements.items()}
        return CargoState(self.ship, self.vehicles, new_placements, self.use_bonus_objective)

    def invalidate_cache(self):
        self._cached_value = None

    def get_grid(self) -> List[List[Optional[str]]]:
        grid = [[None for _ in range(self.ship.w)] for _ in range(self.ship.l)]
        for vid, p in self.placements.items():
            if p.in_ship:
                v = self.vehicles[vid]
                eff_w, eff_l = v.get_effective_dimensions(p.orientation)
                for dy in range(eff_l):
                    for dx in range(eff_w):
                        gx, gy = p.x + dx, p.y + dy
                        if 0 <= gx < self.ship.w and 0 <= gy < self.ship.l:
                            grid[gy][gx] = vid
        return grid

    def get_total_weight(self) -> int:
        return sum(self.vehicles[vid].weight for vid, p in self.placements.items() if p.in_ship)

    def get_total_shipping_fee(self) -> int:
        return sum(self.vehicles[vid].shipping_fee for vid, p in self.placements.items() if p.in_ship)

    def count_violations(self) -> Tuple[int, int, int]:
        grid_count = [[0 for _ in range(self.ship.w)] for _ in range(self.ship.l)]
        out_of_bounds, overlap, total_weight = 0, 0, 0

        for vid, p in self.placements.items():
            if not p.in_ship: continue
            v = self.vehicles[vid]
            total_weight += v.weight
            eff_w, eff_l = v.get_effective_dimensions(p.orientation)

            if p.x < 0 or p.y < 0 or (p.x + eff_w) > self.ship.w or (p.y + eff_l) > self.ship.l:
                for dy in range(eff_l):
                    for dx in range(eff_w):
                        gx, gy = p.x + dx, p.y + dy
                        if gx < 0 or gx >= self.ship.w or gy < 0 or gy >= self.ship.l:
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

        excess_weight = max(0, total_weight - self.ship.max_capacity)
        return out_of_bounds, overlap, excess_weight

    def is_valid(self) -> bool:
        oob, ovl, ex_w = self.count_violations()
        return (oob == 0) and (ovl == 0) and (ex_w == 0)

    def can_place_vehicle(self, vid: str, x: int, y: int, orientation: str, ignore_vids: Optional[set] = None) -> bool:
        ignore_vids = {vid} if ignore_vids is None else set(ignore_vids) | {vid}
        v = self.vehicles[vid]
        eff_w, eff_l = v.get_effective_dimensions(orientation)

        if x < 0 or y < 0 or (x + eff_w) > self.ship.w or (y + eff_l) > self.ship.l: return False

        current_weight = sum(self.vehicles[other_id].weight for other_id, p in self.placements.items() if p.in_ship and other_id not in ignore_vids)
        if current_weight + v.weight > self.ship.max_capacity: return False

        for other_id, p in self.placements.items():
            if not p.in_ship or other_id in ignore_vids: continue
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
    def generate_random_initial_state(cls, ship: Ship, vehicles: Dict[str, Vehicle], use_bonus_objective: bool = False) -> 'CargoState':
        placements = {vid: VehiclePlacement(vid, False, -1, -1, "Horizontal") for vid in vehicles}
        state = cls(ship, vehicles, placements, use_bonus_objective)
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