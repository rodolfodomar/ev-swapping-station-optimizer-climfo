# src/cost_function.py

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist


class CostCalculator:
    """
    Calculates the total annual cost for a given set of EVBSS locations.

    This class implements the multi-objective decision model described in the paper,
    combining investment cost (f1), user time cost (f2), and swapping/queueing
    cost (f3).
    """
    def __init__(self, cost_params: dict):
        """
        Initializes the CostCalculator with all necessary economic and
        operational parameters.

        Args:
            cost_params (dict): A dictionary containing all parameters from
                                Table 1 of the paper.
        """
        self.params = cost_params

    def calculate_total_cost(self, solution_indices: np.ndarray, candidate_locations: np.ndarray, demand_points: np.ndarray) -> float:
        """
        The main public method to calculate the total cost for a given solution.

        Args:
            solution_indices (np.ndarray): An array of integer indices representing
                                           the chosen locations from the candidate list.
            candidate_locations (np.ndarray): A (M x 2) array of coordinates for all
                                              M candidate locations.
            demand_points (np.ndarray): A (P x 3) array containing [x, y, demand] for
                                        all P demand points.

        Returns:
            float: The total annualized cost for the given solution.
        """
        # Ensure indices are integers for array slicing
        solution_indices = np.round(solution_indices).astype(int)

        # Get the coordinates of the chosen station locations
        station_coords = candidate_locations[solution_indices]

        # 1. Assign demand to stations using Voronoi diagrams
        station_demands = self._assign_demand_via_voronoi(station_coords, demand_points)

        # 2. Calculate the three cost components
        f1_investment = self._calculate_f1_investment_cost(len(solution_indices))
        f2_user_time = self._calculate_f2_user_time_cost(station_coords, demand_points)
        f3_swapping_queue = self._calculate_f3_swapping_queue_cost(station_demands)

        # 3. Return the sum
        total_cost = f1_investment + f2_user_time + f3_swapping_queue
        return total_cost

    def _assign_demand_via_voronoi(self, station_coords: np.ndarray, demand_points: np.ndarray) -> np.ndarray:
        """
        Assigns each demand point to the nearest station and calculates the total
        demand for each station.
        """
        demand_coords = demand_points[:, :2]
        demands = demand_points[:, 2]

        # Compute pairwise distances and find the index of the minimum distance
        # for each demand point. This is faster than a Voronoi object for this specific task.
        distances = cdist(demand_coords, station_coords)
        closest_station_indices = np.argmin(distances, axis=1)

        # Sum the demands for each station
        num_stations = station_coords.shape[0]
        station_demands = np.zeros(num_stations)
        for i in range(num_stations):
            station_demands[i] = demands[closest_station_indices == i].sum()

        return station_demands

    def _calculate_f1_investment_cost(self, num_stations: int) -> float:
        """Calculates f1: Annualized Investment and Construction Cost."""
        # Note: This is a simplified version based on the paper's text.
        # A full implementation would use all parameters from Table 1.
        cost_per_evbss = self.params['construction_cost_evbss']
        cost_ccsb = self.params['construction_cost_ccsb'] # Fixed cost, independent of num_stations

        # Annualize costs using a capital recovery factor (simplified here)
        # A more complex model would annualize each component (buildings, machines, batteries)
        # based on its own lifetime and the interest rate.
        annualized_evbss_cost = num_stations * cost_per_evbss / self.params['lifetime']
        annualized_ccsb_cost = cost_ccsb / self.params['lifetime']

        # Placeholder for other costs like batteries, staff, transport vehicles
        # For simplicity, we'll model this as a cost proportional to the number of stations
        annualized_operational_cost = num_stations * self.params['operational_cost_per_station']

        return annualized_evbss_cost + annualized_ccsb_cost + annualized_operational_cost

    def _calculate_f2_user_time_cost(self, station_coords: np.ndarray, demand_points: np.ndarray) -> float:
        """Calculates f2: Annual User Time (Travel) Cost."""
        demand_coords = demand_points[:, :2]
        demands = demand_points[:, 2] # Annual demand from each point

        distances = cdist(demand_coords, station_coords)
        min_distances = np.min(distances, axis=1) # Distance from each demand point to its nearest station

        # Calculate total travel time per year
        # Time = Distance / Speed
        total_travel_time = np.sum( (min_distances / self.params['avg_speed']) * demands )

        # Convert time to cost
        total_time_cost = total_travel_time * self.params['user_time_cost_per_hour']

        return total_time_cost

    def _calculate_f3_swapping_queue_cost(self, station_demands: np.ndarray) -> float:
        """
        Calculates f3: Annual Swapping Service and Queueing Cost.
        This uses the M/M/c queueing model as described in the paper (Eq. 9).
        """
        total_queueing_cost = 0
        for demand in station_demands:
            if demand == 0:
                continue

            # Parameters for M/M/c queue model
            lambda_rate = demand / 365 / 24  # Arrival rate (swaps per hour)
            mu_rate = 60 / self.params['avg_service_time_mins'] # Service rate per BSM (swaps per hour)
            c = self.params['bsm_per_station'] # Number of servers (Battery Swapping Machines)

            # Check for system stability
            if lambda_rate >= c * mu_rate:
                # If the system is unstable, the queue grows infinitely.
                # Assign a very high penalty cost to prevent the optimizer from choosing this solution.
                total_queueing_cost += 1e9 # Large penalty
                continue

            # Calculate rho (traffic intensity)
            rho = lambda_rate / (c * mu_rate)

            # Calculate P0 (probability of zero customers in the system)
            sum_term = np.sum([(c * rho)**n / np.math.factorial(n) for n in range(c)])
            p0_denominator = sum_term + (c * rho)**c / (np.math.factorial(c) * (1 - rho))
            p0 = 1 / p0_denominator

            # Calculate Lq (average number of customers in the queue) - Eq. (9)
            lq = (p0 * (c * rho)**c * rho) / (np.math.factorial(c) * (1 - rho)**2)

            # Calculate Wq (average waiting time in the queue) using Little's Law
            wq = lq / lambda_rate # in hours

            # Annual cost for this station
            station_annual_queue_cost = wq * demand * self.params['user_time_cost_per_hour']
            total_queueing_cost += station_annual_queue_cost

        return total_queueing_cost