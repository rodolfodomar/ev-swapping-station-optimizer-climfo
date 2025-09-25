# src/climfo_optimizer.py

import numpy as np


class CLIMFO:
    """
    Implements the Chaos Levy-flight Moth-flame Optimization (CLIMFO) algorithm.
    """
    def __init__(self, N_moths: int, Max_Iter: int, D: int, lower_bound: int, upper_bound: int):
        """
        Initializes the optimizer.

        Args:
            N_moths (int): Number of moths (population size).
            Max_Iter (int): Maximum number of iterations.
            D (int): Dimension of the problem (number of stations to locate).
            lower_bound (int): The lower bound of the search space (e.g., 0).
            upper_bound (int): The upper bound of the search space (e.g., M-1 for M candidates).
        """
        self.N_moths = N_moths
        self.Max_Iter = Max_Iter
        self.D = D
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def _initialize_population_chaotic(self) -> np.ndarray:
        """
        Initializes the moth population using a chaotic logistic map to ensure
        good diversity and coverage of the search space. (Eq. 10)
        """
        moths = np.zeros((self.N_moths, self.D))
        # Initial vector, must not be 0.25, 0.5, or 0.75
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
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        sigma_v = 1

        u = np.random.normal(0, sigma_u, self.D)
        v = np.random.normal(0, sigma_v, self.D)

        step = u / (np.abs(v)**(1 / beta))

        # The 0.01 factor is a scaling factor to control the step size
        return 0.01 * step


    def optimize(self, cost_function, candidate_locations, demand_points) -> tuple:
        """
        Runs the main optimization loop.

        Args:
            cost_function (CostCalculator): An instantiated cost calculator object.
            candidate_locations (np.ndarray): Array of candidate location coordinates.
            demand_points (np.ndarray): Array of demand point data.

        Returns:
            tuple: (best_solution_indices, best_cost_value, cost_history)
        """
        # 1. Initialization
        moth_positions = self._initialize_population_chaotic()
        cost_history = []

        for iter_num in range(self.Max_Iter):
            print(f"Iteration {iter_num + 1}/{self.Max_Iter}...")
            # 2. Calculate fitness of all moths
            moth_fitness = np.array([
                cost_function.calculate_total_cost(moth, candidate_locations, demand_points)
                for moth in moth_positions
            ])

            # 3. Sort moths and define flames
            sorted_indices = np.argsort(moth_fitness)
            sorted_fitness = moth_fitness[sorted_indices]
            sorted_positions = moth_positions[sorted_indices, :]

            if iter_num == 0:
                flame_positions = sorted_positions
                flame_fitness = sorted_fitness
            else:
                # Combine and re-sort to find the best overall solutions
                combined_pop = np.vstack((flame_positions, moth_positions))
                combined_fit = np.hstack((flame_fitness, moth_fitness))
                
                sorted_combined_indices = np.argsort(combined_fit)
                flame_positions = combined_pop[sorted_combined_indices[:self.N_moths], :]
                flame_fitness = combined_fit[sorted_combined_indices[:self.N_moths]]

            best_solution = flame_positions[0, :]
            best_fitness = flame_fitness[0]
            cost_history.append(best_fitness)

            # 4. Update number of flames (decreases linearly)
            num_flames = round(self.N_moths - iter_num * ((self.N_moths - 1) / self.Max_Iter))

            # 5. Update moth positions
            for i in range(self.N_moths):
                flame_target_index = min(i, num_flames - 1)
                target_flame = flame_positions[flame_target_index, :]

                # MFO logarithmic spiral movement
                distance_to_flame = np.abs(target_flame - moth_positions[i, :])
                b = 1
                t = (iter_num / self.Max_Iter * -1 - 1) * np.random.rand() - 1
                spiral_move = distance_to_flame * np.exp(b * t) * np.cos(2 * np.pi * t) + target_flame

                # CLIMFO enhancement: add Levy flight
                levy_move = self._levy_flight()
                
                moth_positions[i, :] = spiral_move + levy_move

            # 6. Boundary checking
            moth_positions = np.clip(moth_positions, self.lower_bound, self.upper_bound)
            moth_positions = np.round(moth_positions).astype(int)

        return best_solution, best_fitness, cost_history