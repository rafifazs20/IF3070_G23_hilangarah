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

    def schedule(self, t: int) -> float:
        return self.initial_temperature * (self.cooling_rate ** t)
    
    def run(self, initial_state: CargoState) -> Tuple[CargoState, Dict[str, Any]]:
        current_state = initial_state
        current_value = current_state.evaluate()
        
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
            else:
                exponent = delta_e / temperature
                if exponent < -700:
                    acceptance_probability = 0.0
                else:
                    acceptance_probability = math.exp(exponent)
                
                if random.random() < acceptance_probability:
                    current_state = next_state
                    current_value = next_value
        
        return current_state, {}    