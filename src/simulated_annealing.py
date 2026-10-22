import math
import random
import time
from typing import Tuple, Dict, Any, List
from src.models import CargoState

class SimulatedAnnealing:
    def __init__(
        self, 
        initial_temperature: float = 1000.0, 
        cooling_rate: float = 0.99, 
        min_temperature: float = 1e-3,
        max_iterations: int = 5000
    ):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations

    # Fungsi cooling schedule pakai T(t) = T0 * (cooling_rate ^ t)
    def schedule(self, t: int) -> float:
        return self.initial_temperature * (self.cooling_rate ** t)

    # Fungsi untuk menjalankan algoritma dari initial_state, lalu mengembalikan final_state dan metrics
    def run(self, initial_state: CargoState) -> Tuple[CargoState, Dict[str, Any]]:
        current_state = initial_state
        current_value = current_state.evaluate()
        
        history_obj: List[float] = [current_value]
        history_prob: List[float] = [] # menyimpan nilai e^(Delta E / T) 

        # Variabel untuk menghitung stuck frequency
        stuck_frequency = 0
        consecutive_rejections = 0
        
        start_time = time.time()
        
        for t in range(self.max_iterations):
            temperature = self.schedule(t)
            if temperature < self.min_temperature:
                break
                
            next_state = current_state.get_random_neighbor()
            next_value = next_state.evaluate()
            delta_e = next_value - current_value
            
            if delta_e > 0:
                current_state = next_state
                current_value = next_value
                history_prob.append(1.0)
                consecutive_rejections = 0  
            else:
                exponent = delta_e / temperature
                if exponent < -700:
                    acceptance_probability = 0.0
                else:
                    acceptance_probability = math.exp(exponent)
                
                history_prob.append(acceptance_probability)
                
                if random.random() < acceptance_probability:
                    current_state = next_state
                    current_value = next_value
                    consecutive_rejections = 0  
                else:
                    consecutive_rejections += 1
                    if consecutive_rejections > 20:
                        stuck_frequency += 1
                        consecutive_rejections = 0  
            
            history_obj.append(current_value)
            
        end_time = time.time()

        # dictionary metrics
        metrics = {
            "initial_value": initial_state.evaluate(),
            "final_value": current_value,
            "duration_seconds": end_time - start_time,
            "iterations_ran": len(history_obj) - 1,
            "history_obj": history_obj,
            "history_prob": history_prob,
            "stuck_frequency": stuck_frequency,  
            "is_valid": current_state.is_valid()
        }
        
        return current_state, metrics