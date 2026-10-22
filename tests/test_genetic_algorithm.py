from src.dataset import get_default_problem
from src.genetic_algorithm import GeneticAlgorithm

def main():
    print("=== Full Test: Genetic Algorithm ===")
    ship, vehicles = get_default_problem()
    
    print(f"Dimensi Kapal: {ship.w}x{ship.l} | Max Kapasitas: {ship.max_capacity}")
    print(f"Jumlah Kendaraan: {len(vehicles)}")
    
    ga = GeneticAlgorithm(
        ship=ship,
        vehicles=vehicles,
        pop_size=50,
        mutation_rate=0.2,
        crossover_rate=0.8,
        max_generations=50,
        use_bonus_objective=False
    )
    
    print("\nMenjalankan Evolusi...")
    best_state, hist_max, hist_avg = ga.run()
    
    print("\n=== HASIL TERBAIK ===")
    print(f"Total Shipping Fee: {best_state.get_total_shipping_fee()}")
    print(f"Total Berat: {best_state.get_total_weight()} / {ship.max_capacity}")
    
    oob, ovl, ex_w = best_state.count_violations()
    print(f"Pelanggaran -> Out of Bounds: {oob}, Overlap: {ovl}, Overweight: {ex_w}")
    
    if best_state.is_valid():
        print("Status Akhir: VALID")
    else:
        print("Status Akhir: TIDAK VALID")

if __name__ == "__main__":
    main()