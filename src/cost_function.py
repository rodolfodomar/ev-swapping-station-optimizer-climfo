# src/cost_function.py

import math  # <--- ADD THIS LINE

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist


class CostCalculator:
    """
    Calculates the total annual cost for a given set of EVBSS locations.
    """
    def __init__(self, cost_params: dict):
        # ... (this method is unchanged)
        self.params = cost_params

    def calculate_total_cost(self, solution_indices: np.ndarray, candidate_locations: np.ndarray, demand_points: np.ndarray) -> float:
        # ... (this method is unchanged)
        solution_indices = np.round(solution_indices).astype(int)
        station_coords = candidate_locations[solution_indices]
        station_demands = self._assign_demand_via_voronoi(station_coords, demand_points)
        f1_investment = self._calculate_f1_investment_cost(len(solution_indices))
        f2_user_time = self._calculate_f2_user_time_cost(station_coords, demand_points)
        f3_swapping_queue = self._calculate_f3_swapping_queue_cost(station_demands)
        total_cost = f1_investment + f2_user_time + f3_swapping_queue
        return total_cost

    def _assign_demand_via_voronoi(self, station_coords: np.ndarray, demand_points: np.ndarray) -> np.ndarray:
        # ... (this method is unchanged)
        demand_coords = demand_points[:, :2]
        demands = demand_points[:, 2]
        distances = cdist(demand_coords, station_coords)
        closest_station_indices = np.argmin(distances, axis=1)
        num_stations = station_coords.shape[0]
        station_demands = np.zeros(num_stations)
        for i in range(num_stations):
            station_demands[i] = demands[closest_station_indices == i].sum()
        return station_demands

    def _calculate_f1_investment_cost(self, num_stations: int) -> float:
        # ... (this method is unchanged)
        cost_per_evbss = self.params['construction_cost_evbss']
        cost_ccsb = self.params['construction_cost_ccsb']
        annualized_evbss_cost = num_stations * cost_per_evbss / self.params['lifetime']
        annualized_ccsb_cost = cost_ccsb / self.params['lifetime']
        annualized_operational_cost = num_stations * self.params['operational_cost_per_station']
        return annualized_evbss_cost + annualized_ccsb_cost + annualized_operational_cost

    def _calculate_f2_user_time_cost(self, station_coords: np.ndarray, demand_points: np.ndarray) -> float:
        # ... (this method is unchanged)
        demand_coords = demand_points[:, :2]
        demands = demand_points[:, 2]
        distances = cdist(demand_coords, station_coords)
        min_distances = np.min(distances, axis=1)
        total_travel_time = np.sum( (min_distances / self.params['avg_speed']) * demands )
        total_time_cost = total_travel_time * self.params['user_time_cost_per_hour']
        return total_time_cost

    def _calculate_f3_swapping_queue_cost(self, station_demands: np.ndarray) -> float:
        """
        Calculates f3: Annual Swapping Service and Queueing Cost.
        """
        total_queueing_cost = 0
        for demand in station_demands:
            if demand == 0:
                continue
            lambda_rate = demand / 365 / 24
            mu_rate = 60 / self.params['avg_service_time_mins']
            c = self.params['bsm_per_station']
            if lambda_rate >= c * mu_rate:
                total_queueing_cost += 1e9
                continue
            rho = lambda_rate / (c * mu_rate)
            # CORRECTED SECTION
            sum_term = np.sum([(c * rho)**n / math.factorial(n) for n in range(c)])
            p0_denominator = sum_term + (c * rho)**c / (math.factorial(c) * (1 - rho))
            p0 = 1 / p0_denominator
            lq = (p0 * (c * rho)**c * rho) / (math.factorial(c) * (1 - rho)**2)
            wq = lq / lambda_rate
            station_annual_queue_cost = wq * demand * self.params['user_time_cost_per_hour']
            total_queueing_cost += station_annual_queue_cost
        return total_queueing_cost