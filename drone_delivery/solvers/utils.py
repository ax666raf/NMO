import json
import math
import numpy as np

class DroneInstance:
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.instance_id = data["instance_id"]
        self.n_customers = data["n_customers"]
        self.n_drones = data["n_drones"]
        self.depot = data["depot"]
        self.customers = data["customers"]
        self.demands = data["demands"]
        self.battery_km = data["battery_km"]
        self.payload_kg = data["payload_kg"]
        self.no_fly_zones = data.get("no_fly_zones", [])
        
        # Build distance matrix (index 0 is depot, 1..n are customers)
        self.n_nodes = self.n_customers + 1
        self.nodes = [self.depot] + self.customers
        self.distance_matrix = np.zeros((self.n_nodes, self.n_nodes))
        
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    self.distance_matrix[i][j] = math.hypot(
                        self.nodes[i][0] - self.nodes[j][0],
                        self.nodes[i][1] - self.nodes[j][1]
                    )

    def energy_cost(self, i, j):
        # Alpha=1.0, Beta=0.0 => proportional to distance
        return self.distance_matrix[i][j]

    def get_demand(self, i):
        # Demand of depot is 0, customers have index i-1 in demands list
        if i == 0:
            return 0.0
        return self.demands[i-1]

def evaluate_solution(instance, routes):
    """
    Evaluates a solution given a list of routes.
    Each route is a list of node indices starting and ending at 0.
    Returns (feasible, cost, route_costs, route_demands)
    """
    total_cost = 0.0
    feasible = True
    route_costs = []
    route_demands = []
    visited_customers = set()

    for route in routes:
        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            feasible = False
            
        route_dist = 0.0
        route_demand = 0.0
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            route_dist += instance.distance_matrix[u][v]
            if v != 0:
                route_demand += instance.get_demand(v)
                visited_customers.add(v)
                
        route_costs.append(route_dist)
        route_demands.append(route_demand)
        total_cost += route_dist
        
        if route_dist > instance.battery_km + 1e-6:
            feasible = False
        if route_demand > instance.payload_kg + 1e-6:
            feasible = False
            
    # Check if all customers were visited exactly once
    if len(visited_customers) != instance.n_customers:
        feasible = False
        
    return feasible, total_cost, route_costs, route_demands
