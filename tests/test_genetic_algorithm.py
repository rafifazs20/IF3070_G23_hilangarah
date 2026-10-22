from src.dataset import get_default_problem
from src.genetic_algorithm import GeneticAlgorithm

def main():
    print("--- Testing (Inisialisasi, Fitness, & Seleksi) ---")
    ship, vehicles = get_default_problem()
    
    ga = GeneticAlgorithm(ship, vehicles, pop_size=10)
    
    print("\n1. Test Inisialisasi...")
    population = ga._initialize_population()
    print(f"Berhasil membuat {len(population)} individu acak.")
    
    print("\n2. Test Fitness...")
    for i in range(5):
        state = population[i]
        fit = ga._calculate_fitness(state)
        oob, ovl, ex_w = state.count_violations()
        valid_status = "VALID" if state.is_valid() else "TIDAK VALID"
        print(f"Individu {i+1} | Status: {valid_status} | Fitness: {fit} | OOB: {oob}, Overlap: {ovl}, Overweight: {ex_w}")
        
    print("\n3. Test Seleksi...")
    for i in range(3):
        selected = ga._roulette_wheel_selection(population)
        fit = ga._calculate_fitness(selected)
        print(f"Seleksi ke-{i+1} terpilih individu dengan fitness: {fit}")
        
    print("\nTesting selesai!")

if __name__ == "__main__":
    main()