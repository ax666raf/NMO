import sys
import os
import random
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from utils import DroneInstance, evaluate_solution
from branch_and_bound import greedy_nearest_neighbor

def get_neighborhood(routes):
    neighbors = []
    
    # 1. Relocate: move customer from one route to another
    for r1_idx, r1 in enumerate(routes):
        for i_idx in range(1, len(r1) - 1): # exclude depot
            customer = r1[i_idx]
            
            for r2_idx, r2 in enumerate(routes):
                if r1_idx == r2_idx: continue
                
                # Insert at every possible position in r2
                for insert_pos in range(1, len(r2)):
                    new_r1 = r1[:i_idx] + r1[i_idx+1:]
                    new_r2 = r2[:insert_pos] + [customer] + r2[insert_pos:]
                    
                    new_routes = [list(r) for r in routes]
                    new_routes[r1_idx] = new_r1
                    new_routes[r2_idx] = new_r2
                    
                    neighbors.append({
                        'routes': new_routes,
                        'move': ('relocate', customer, r1_idx, r2_idx)
                    })
                    
    # 2. Swap: exchange customer between routes
    for r1_idx, r1 in enumerate(routes):
        for r2_idx, r2 in enumerate(routes):
            if r1_idx >= r2_idx: continue # Avoid duplicates
            
            for i_idx in range(1, len(r1) - 1):
                for j_idx in range(1, len(r2) - 1):
                    c1 = r1[i_idx]
                    c2 = r2[j_idx]
                    
                    new_routes = [list(r) for r in routes]
                    new_routes[r1_idx][i_idx] = c2
                    new_routes[r2_idx][j_idx] = c1
                    
                    neighbors.append({
                        'routes': new_routes,
                        'move': ('swap', c1, c2, r1_idx, r2_idx)
                    })
                    
    # 3. 2-opt intra-route
    for r_idx, r in enumerate(routes):
        for i in range(1, len(r) - 2):
            for j in range(i + 1, len(r) - 1):
                new_r = r[:i] + r[i:j+1][::-1] + r[j+1:]
                new_routes = [list(r) for r in routes]
                new_routes[r_idx] = new_r
                
                neighbors.append({
                    'routes': new_routes,
                    'move': ('2opt', r_idx, i, j)
                })
                
    return neighbors

def solve_tabu(instance: DroneInstance, max_iter=500, max_no_improve=200, tabu_tenure=7, time_limit=30):
    start_time = time.time()
    _, initial_routes = greedy_nearest_neighbor(instance)
    
    if initial_routes is None:
        return {'feasible': False, 'cost': float('inf'), 'runtime': 0, 'method': 'TS'}

    best_known_routes = initial_routes
    valid, best_known_cost, _, _ = evaluate_solution(instance, best_known_routes)
    
    if not valid:
        best_known_cost = float('inf')
        
    curr_routes = initial_routes
    curr_cost = best_known_cost
    
    tabu_list = {} # move signature -> iter available
    
    iter_best = best_known_cost
    no_improve_iters = 0
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        neighbors = get_neighborhood(curr_routes)
        
        best_neighborhood_cost = float('inf')
        best_neighbor = None
        best_move = None
        
        for neighbor in neighbors:
            valid, cost, _, _ = evaluate_solution(instance, neighbor['routes'])
            if not valid: continue
            
            move = neighbor['move']
            is_tabu = False
            
            # Simple Tabu signature representation
            sig = None
            if move[0] == 'relocate':
                sig = ('relocate', move[1], move[3]) # Moving customer to new drone
            elif move[0] == 'swap':
                sig = ('swap', move[1], move[2])
            elif move[0] == '2opt':
                sig = ('2opt', move[1], move[2], move[3])
                
            if sig in tabu_list and tabu_list[sig] > iteration:
                is_tabu = True
                
            # Aspiration criterion
            if is_tabu and cost < best_known_cost:
                is_tabu = False
                
            if not is_tabu and cost < best_neighborhood_cost:
                best_neighborhood_cost = cost
                best_neighbor = neighbor['routes']
                best_move = sig
                
        if best_neighbor is None:
            break # No valid neighbors
            
        curr_routes = best_neighbor
        curr_cost = best_neighborhood_cost
        
        if best_move is not None:
            # Add to tabu list
            if best_move[0] == 'relocate':
                # Tabu moving back
                reverse_sig = ('relocate', move[1], move[2])
                tabu_list[reverse_sig] = iteration + tabu_tenure
            elif best_move[0] == 'swap':
                tabu_list[best_move] = iteration + tabu_tenure
            elif best_move[0] == '2opt':
                tabu_list[best_move] = iteration + tabu_tenure
                
        if curr_cost < best_known_cost:
            best_known_cost = curr_cost
            best_known_routes = curr_routes
            no_improve_iters = 0
        else:
            no_improve_iters += 1
            
        if no_improve_iters >= max_no_improve:
            break

    return {
        'feasible': best_known_cost != float('inf'),
        'cost': best_known_cost,
        'runtime': round(time.time() - start_time, 2),
        'gap': '-',
        'method': 'TS'
    }

if __name__ == "__main__":
    example_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'instances', 'small', 'small_01.json')
    if os.path.exists(example_path):
        inst = DroneInstance(example_path)
        print("Running Tabu Search...")
        res = solve_tabu(inst, time_limit=30)
        print("Result:", res)
