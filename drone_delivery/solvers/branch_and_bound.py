import pulp
import sys
import os
import time
import heapq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from formulation1_vrp import build_vrp_model
from utils import DroneInstance, evaluate_solution

def greedy_nearest_neighbor(instance: DroneInstance):
    # Initialize routes
    routes = []
    unvisited = set(range(1, instance.n_customers + 1))
    
    while unvisited and len(routes) < instance.n_drones:
        curr_route = [0]
        curr_dist = 0.0
        curr_demand = 0.0
        
        while unvisited:
            curr_node = curr_route[-1]
            # Find nearest feasible
            nearest = None
            min_dist = float('inf')
            for v in unvisited:
                dist = instance.distance_matrix[curr_node][v]
                return_dist = instance.distance_matrix[v][0]
                demand = instance.get_demand(v)
                
                if (curr_dist + dist + return_dist <= instance.battery_km) and (curr_demand + demand <= instance.payload_kg):
                    if dist < min_dist:
                        min_dist = dist
                        nearest = v
                        
            if nearest is not None:
                curr_route.append(nearest)
                curr_dist += min_dist
                curr_demand += instance.get_demand(nearest)
                unvisited.remove(nearest)
            else:
                break
                
        curr_route.append(0) # Return to depot
        routes.append(curr_route)
        
    # If not all visited, return None
    if unvisited:
        # Simple fallback for bounds if needed: assign rest to depot to get valid nodes (though infeasible)
        pass 

    feasible, total_cost, _, _ = evaluate_solution(instance, routes)
    if feasible:
        return total_cost, routes
    return float('inf'), None

class Node:
    def __init__(self, fixed_vars, lower_bound):
        self.fixed_vars = fixed_vars  # dict {(i,j,k): 0 or 1}
        self.lower_bound = lower_bound

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

def solve_bnb(instance: DroneInstance, time_limit=120):
    start_time = time.time()
    
    # Get initial greedy solution to bound
    best_obj, best_routes = greedy_nearest_neighbor(instance)
    
    # Build model (relaxing binaries by setting category to continuous)
    model, x_vars, u_vars = build_vrp_model(instance)
    for var in x_vars.values():
        var.cat = pulp.LpContinuous
        var.lowBound = 0
        var.upBound = 1
        
    def solve_relaxation(node_fixed_vars):
        # We need to temporarily add constraints for fixed vars, solve, then remove
        constraints_added = []
        for (i, j, k), val in node_fixed_vars.items():
            name = f"Fixed_{i}_{j}_{k}"
            # Check if exists (which might if we don't clear properly)
            if name in model.constraints:
                del model.constraints[name]
            constr = x_vars[i, j, k] == val
            model.addConstraint(constr, name)
            constraints_added.append(name)
            
        # Suppress logging
        model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        status = model.status
        obj_val = pulp.value(model.objective)
        
        # Gather variable values
        var_values = {}
        if status == pulp.LpStatusOptimal:
            for (i, j, k), var in x_vars.items():
                var_values[i, j, k] = var.varValue
                
        # Remove temporarily added constraints
        for name in constraints_added:
            del model.constraints[name]
            
        return status, obj_val, var_values

    # Root node
    root_status, root_obj, root_vals = solve_relaxation({})
    if root_status != pulp.LpStatusOptimal:
        return {'feasible': False, 'cost': float('inf'), 'runtime': time.time() - start_time, 'gap': 0, 'method': 'BnB'}
        
    pq = []
    heapq.heappush(pq, Node({}, root_obj))
    best_lower_bound = root_obj
    
    while pq:
        if time.time() - start_time > time_limit:
            break
            
        curr_node = heapq.heappop(pq)
        
        # Prune by bound
        if curr_node.lower_bound >= best_obj:
            continue
            
        # Solve again to get variable assignments
        status, obj_val, var_values = solve_relaxation(curr_node.fixed_vars)
        if status != pulp.LpStatusOptimal or obj_val >= best_obj:
            continue
            
        # Update best minimum lower bound of remaining nodes
        if pq:
            best_lower_bound = min(best_lower_bound, pq[0].lower_bound)
        else:
            best_lower_bound = obj_val
            
        # Check integer feasibility
        fractional_vars = {}
        is_integer = True
        for key, val in var_values.items():
            if val is None:
                continue
            if 1e-5 < val < 1 - 1e-5:
                is_integer = False
                fractional_vars[key] = min(val, 1 - val) # Distance to nearest int
                
        if is_integer:
            if obj_val < best_obj:
                best_obj = obj_val
            continue
            
        # Branch on most fractional variable
        # closest to 0.5 means min distance to int is highest
        branch_var = max(fractional_vars.items(), key=lambda item: item[1])[0]
        
        # Two children
        left_fixed = curr_node.fixed_vars.copy()
        left_fixed[branch_var] = 1
        heapq.heappush(pq, Node(left_fixed, obj_val))
        
        right_fixed = curr_node.fixed_vars.copy()
        right_fixed[branch_var] = 0
        heapq.heappush(pq, Node(right_fixed, obj_val))
        
    gap = 0
    if best_lower_bound > 0 and best_obj != float('inf'):
        gap = (best_obj - best_lower_bound) / best_lower_bound * 100
        
    return {
        'feasible': best_obj != float('inf'),
        'cost': best_obj,
        'lower_bound': best_lower_bound,
        'runtime': round(time.time() - start_time, 2),
        'gap': round(gap, 2),
        'method': 'BnB'
    }

if __name__ == "__main__":
    example_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'instances', 'small', 'small_01.json')
    if os.path.exists(example_path):
        inst = DroneInstance(example_path)
        print("Running B&B...")
        res = solve_bnb(inst, time_limit=30)
        print("Result:", res)
