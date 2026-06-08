import pulp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'solvers')))
from utils import DroneInstance

def build_flow_model(instance: DroneInstance):
    # Initialize the LP problem
    model = pulp.LpProblem("Drone_Delivery_Flow", pulp.LpMinimize)
    
    n = instance.n_customers
    K = instance.n_drones
    
    # We duplicate the depot: 0 as source, n+1 as sink
    source = 0
    sink = n + 1
    nodes = list(range(1, n + 1)) + [source, sink]
    customers = list(range(1, n + 1))
    drones = list(range(K))
    
    # We extend the distance matrix conceptually to handle sink (which is physically at depot)
    def dist(i, j):
        idx_i = 0 if i == sink else i
        idx_j = 0 if j == sink else j
        return instance.distance_matrix[idx_i][idx_j]
        
    def energy(i, j):
        return dist(i, j)
        
    def demand(i):
        if i == source or i == sink:
            return 0.0
        return instance.get_demand(i)

    # Valid arcs: source to customer, customer to customer, customer to sink, source to sink
    arcs = []
    for i in nodes:
        for j in nodes:
            if i != j and j != source and i != sink:
                arcs.append((i, j))
                
    # Decision Variables
    f = pulp.LpVariable.dicts("f",
                              ((i, j, k) for (i, j) in arcs for k in drones),
                              lowBound=0,
                              cat='Continuous')
                              
    y = pulp.LpVariable.dicts("y",
                              arcs,
                              cat='Binary')
                              
    # Objective
    model += pulp.lpSum(energy(i, j) * y[i, j] for (i, j) in arcs), "Total_Energy"
    
    # Constraints
    # 1. Flow conservation
    for k in drones:
        for i in nodes:
            out_flow = pulp.lpSum(f[i, j, k] for j in nodes if (i, j) in arcs)
            in_flow = pulp.lpSum(f[j, i, k] for j in nodes if (j, i) in arcs)
            
            if i == source:
                model += out_flow - in_flow == 1, f"Flow_source_{k}"
            elif i == sink:
                model += out_flow - in_flow == -1, f"Flow_sink_{k}"
            else:
                model += out_flow - in_flow == 0, f"Flow_cust_{i}_{k}"

    # 2. Link flow to binary arc usage
    for (i, j) in arcs:
        model += pulp.lpSum(f[i, j, k] for k in drones) <= K * y[i, j], f"Link_flow_usage_{i}_{j}"

    # 3. Payload constraint
    for k in drones:
        model += pulp.lpSum(
            demand(i) * f[source, i, k] 
            for i in customers
        ) <= instance.payload_kg, f"Payload_{k}"

    # 4. Battery constraint
    for k in drones:
        model += pulp.lpSum(
            dist(i, j) * f[i, j, k] 
            for (i, j) in arcs
        ) <= instance.battery_km, f"Battery_{k}"

    # 5. Each customer served exactly once
    for i in customers:
        model += pulp.lpSum(f[i, j, k] for j in nodes if (i, j) in arcs for k in drones) == 1, f"Visit_{i}"
        
    return model, f, y

if __name__ == "__main__":
    example_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'instances', 'small', 'small_01.json')
    if os.path.exists(example_path):
        inst = DroneInstance(example_path)
        model, f, y = build_flow_model(inst)
        print(f"Model Flow built successfully with {len(model.variables())} variables and {len(model.constraints)} constraints.")
