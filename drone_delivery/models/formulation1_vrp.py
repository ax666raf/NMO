import pulp
import sys
import os

# Add solvers directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'solvers')))
from utils import DroneInstance

def build_vrp_model(instance: DroneInstance):
    # Initialize the LP problem
    model = pulp.LpProblem("Drone_Delivery_VRP", pulp.LpMinimize)
    
    n = instance.n_customers
    K = instance.n_drones
    nodes = list(range(n + 1))
    customers = list(range(1, n + 1))
    drones = list(range(K))
    
    # Decision Variables
    # x_ijk = 1 if drone k travels from i to j
    x = pulp.LpVariable.dicts("x", 
                              ((i, j, k) for i in nodes for j in nodes for k in drones if i != j),
                              cat='Binary')
                              
    # u_ik = position of customer i in the route of drone k
    u = pulp.LpVariable.dicts("u",
                              ((i, k) for i in customers for k in drones),
                              lowBound=1, upBound=n, cat='Continuous')
                              
    # Objective: Minimize total energy consumption (proportional to distance)
    model += pulp.lpSum(
        instance.energy_cost(i, j) * x[i, j, k] 
        for i in nodes for j in nodes for k in drones if i != j
    ), "Total_Energy"
    
    # Constraints
    # 1. Each customer is visited exactly once
    for i in customers:
        model += pulp.lpSum(x[i, j, k] for j in nodes for k in drones if i != j) == 1, f"Visit_{i}"
        
    # 2. Flow conservation
    for k in drones:
        for i in nodes:
            out_flow = pulp.lpSum(x[i, j, k] for j in nodes if i != j)
            in_flow = pulp.lpSum(x[j, i, k] for j in nodes if i != j)
            model += out_flow == in_flow, f"Flow_conservation_{i}_{k}"
            
    # 3. Each drone starts and ends at the depot
    for k in drones:
        model += pulp.lpSum(x[0, j, k] for j in customers) == 1, f"Start_depot_{k}"
        model += pulp.lpSum(x[i, 0, k] for i in customers) == 1, f"End_depot_{k}"
        
    # 4. Payload capacity per drone
    for k in drones:
        model += pulp.lpSum(
            instance.get_demand(i) * x[i, j, k] 
            for i in customers for j in nodes if i != j
        ) <= instance.payload_kg, f"Payload_{k}"
        
    # 5. Battery capacity per drone
    for k in drones:
        model += pulp.lpSum(
            instance.energy_cost(i, j) * x[i, j, k] 
            for i in nodes for j in nodes if i != j
        ) <= instance.battery_km, f"Battery_{k}"
        
    # 6. Sub-tour elimination (MTZ)
    for k in drones:
        for i in customers:
            for j in customers:
                if i != j:
                    model += u[i, k] - u[j, k] + n * x[i, j, k] <= n - 1, f"MTZ_{i}_{j}_{k}"
                    
    return model, x, u

if __name__ == "__main__":
    # Test model building with a small instance
    example_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'instances', 'small', 'small_01.json')
    if os.path.exists(example_path):
        inst = DroneInstance(example_path)
        model, x, u = build_vrp_model(inst)
        print(f"Model VRP built successfully with {len(model.variables())} variables and {len(model.constraints)} constraints.")
