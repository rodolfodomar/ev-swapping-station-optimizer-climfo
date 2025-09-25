# src/climfo_optimizer.py

import math

import numpy as np


class CLIMFO:
    """
    Implements the Chaos Levy-flight Moth-flame Optimization (CLIMFO) algorithm.
    """
    def __init__(self, N_moths: int, Max_Iter: int, D: int, lower_bound: int, upper_bound: int):
        # ... (rest of the __init__ method is unchanged)
        self.N_moths = N_moths
        self.Max_Iter = Max_Iter
        self.D = D
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def _initialize_population_chaotic(self) -> np.ndarray:
        # ... (this method is unchanged)
        moths = np.zeros((self.N_moths, self.D))
        chaotic_vector = np.random.uniform(0, 1, self.D)
        while any(x in [0.25, 0.5, 0.75] for x in chaotic_vector):
            chaotic_vector = np.random.uniform(0, 1, self.D)
        for i in range(self.N_moths):
            chaotic_vector = 4 * chaotic_vector * (1 - chaotic_vector)
            moths[i, :] = self.lower_bound + chaotic_vector * (self.upper_bound - self.lower_bound)
        return np.round(moths).astype(int)


    def _levy_flight(self, beta: float = 1.5) -> np.ndarray:
        """
        Generates a step size using a Levy-flight distribution to enhance
        global search capabilities. (Eq. 11)
        """
        # CORRECTED SECTION
        sigma_u = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        sigma_v = 1

        u = np.random.normal(0, sigma_u, self.D)
        v = np.random.normal(0, sigma_v, self.D)

        step = u / (np.abs(v)**(1 / beta))
        return 0.01 * step


    def optimize(self, cost_function, candidate_locations, demand_points) -> tuple:
        # ... (rest of the optimize method is unchanged)
        moth_positions = self._initialize_population_chaotic()
        cost_history = []
        for iter_num in range(self.Max_Iter):
            print(f"Iteration {iter_num + 1}/{self.Max_Iter}...")
            moth_fitness = np.array([
                cost_function.calculate_total_cost(moth, candidate_locations, demand_points)
                for moth in moth_positions
            ])
            sorted_indices = np.argsort(moth_fitness)
            sorted_fitness = moth_fitness[sorted_indices]
            sorted_positions = moth_positions[sorted_indices, :]
            if iter_num == 0:
                flame_positions = sorted_positions
                flame_fitness = sorted_fitness
            else:
                combined_pop = np.vstack((flame_positions, moth_positions))
                combined_fit = np.hstack((flame_fitness, moth_fitness))
                sorted_combined_indices = np.argsort(combined_fit)
                flame_positions = combined_pop[sorted_combined_indices[:self.N_moths], :]
                flame_fitness = combined_fit[sorted_combined_indices[:self.N_moths]]
            best_solution = flame_positions[0, :]
            best_fitness = flame_fitness[0]
            cost_history.append(best_fitness)
            num_flames = round(self.N_moths - iter_num * ((self.N_moths - 1) / self.Max_Iter))
            for i in range(self.N_moths):
                flame_target_index = min(i, num_flames - 1)
                target_flame = flame_positions[flame_target_index, :]
                distance_to_flame = np.abs(target_flame - moth_positions[i, :])
                b = 1
                t = (iter_num / self.Max_Iter * -1 - 1) * np.random.rand() - 1
                spiral_move = distance_to_flame * np.exp(b * t) * np.cos(2 * np.pi * t) + target_flame
                levy_move = self._levy_flight()
                moth_positions[i, :] = spiral_move + levy_move
            moth_positions = np.clip(moth_positions, self.lower_bound, self.upper_bound)
            moth_positions = np.round(moth_positions).astype(int)
        return best_solution, best_fitness, cost_history