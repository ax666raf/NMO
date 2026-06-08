import sys
import os
import random
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from utils import DroneInstance, evaluate_solution

class Individual:
    def __init__(self, chromosome, instance):
        self.chromosome = chromosome
        self.instance = instance
        self.routes = []
        self.cost = float('inf')
        self.feasible = False
        self.decode()
        
    def decode(self):
        self.routes = []
        curr_route = [0]
        curr_dist = 0.0
        curr_demand = 0.0
        
        for c in self.chromosome:
            dist = self.instance.distance_matrix[curr_route[-1]][c]
            return_dist = self.instance.distance_matrix[c][0]
            demand = self.instance.get_demand(c)
            
            if curr_dist + dist + return_dist <= self.instance.battery_km and curr_demand + demand <= self.instance.payload_kg:
                curr_route.append(c)
                curr_dist += dist
                curr_demand += demand
            else:
                curr_route.append(0)
                self.routes.append(curr_route)
                curr_route = [0, c]
                curr_dist = self.instance.distance_matrix[0][c]
                curr_demand = demand
                
        curr_route.append(0)
        self.routes.append(curr_route)
        
        # Repair operator if > K drones
        self.repair()
        
        self.feasible, self.cost, _, _ = evaluate_solution(self.instance, self.routes)
        
    def repair(self):
        # Move most expensive customer from overloaded routes to nearest drones
        while len(self.routes) > self.instance.n_drones:
            # Overloaded route is the last one
            overloaded = self.routes.pop()
            customers_to_insert = [c for c in overloaded if c != 0]
            
            for c in customers_to_insert:
                best_r = -1
                best_pos = -1
                best_ins_cost = float('inf')
                c_demand = self.instance.get_demand(c)
                
                for r_idx in range(len(self.routes)):
                    r = self.routes[r_idx]
                    
                    # Current load and dist
                    curr_load = sum(self.instance.get_demand(n) for n in r)
                    if curr_load + c_demand > self.instance.payload_kg:
                        continue
                        
                    curr_dist = sum(self.instance.distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1))
                    
                    for pos in range(1, len(r)):
                        prev_node = r[pos-1]
                        next_node = r[pos]
                        
                        added_dist = self.instance.distance_matrix[prev_node][c] + self.instance.distance_matrix[c][next_node] - self.instance.distance_matrix[prev_node][next_node]
                        
                        if curr_dist + added_dist <= self.instance.battery_km:
                            if added_dist < best_ins_cost:
                                best_ins_cost = added_dist
                                best_r = r_idx
                                best_pos = pos
                                
                if best_r != -1:
                    self.routes[best_r].insert(best_pos, c)
                else:
                    # Can't fit: leave it as a new route (will marked infeasible)
                    self.routes.append([0, c, 0])
                    break
                    
def crossover_ox(p1, p2):
    size = len(p1)
    start, end = sorted(random.sample(range(size), 2))
    
    child = [-1] * size
    child[start:end+1] = p1[start:end+1]
    
    p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            while p2[p2_idx] in child:
                p2_idx += 1
            child[i] = p2[p2_idx]
            
    return child

def solve_ga(instance: DroneInstance, pop_size=50, max_gen=200, p_cross=0.8, p_mut=0.1, time_limit=60):
    start_time = time.time()
    
    customers = list(range(1, instance.n_customers + 1))
    population = []
    
    for _ in range(pop_size):
        chrom = customers.copy()
        random.shuffle(chrom)
        population.append(Individual(chrom, instance))
        
    best_ind = min(population, key=lambda ind: ind.cost)
    
    for gen in range(max_gen):
        if time.time() - start_time > time_limit:
            break
            
        new_population = []
        
        # Elitism
        new_population.append(min(population, key=lambda ind: ind.cost))
        
        while len(new_population) < pop_size:
            # Tournament Select
            parents = []
            for _ in range(2):
                tournament = random.sample(population, 3)
                parents.append(min(tournament, key=lambda ind: ind.cost))
                
            p1, p2 = parents[0].chromosome, parents[1].chromosome
            
            if random.random() < p_cross:
                child_chrom = crossover_ox(p1, p2)
            else:
                child_chrom = p1.copy()
                
            # Mutation (Swap)
            if random.random() < p_mut:
                i, j = random.sample(range(len(child_chrom)), 2)
                child_chrom[i], child_chrom[j] = child_chrom[j], child_chrom[i]
                
            # Mutation (2-opt)
            if random.random() < 0.05:
                i, j = sorted(random.sample(range(len(child_chrom)), 2))
                child_chrom[i:j+1] = reversed(child_chrom[i:j+1])
                
            new_population.append(Individual(child_chrom, instance))
            
        population = new_population
        current_best = min(population, key=lambda ind: ind.cost)
        if current_best.cost < best_ind.cost:
            best_ind = current_best

    return {
        'feasible': best_ind.feasible,
        'cost': best_ind.cost,
        'runtime': round(time.time() - start_time, 2),
        'gap': '-',
        'method': 'GA'
    }

if __name__ == "__main__":
    example_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'instances', 'small', 'small_01.json')
    if os.path.exists(example_path):
        inst = DroneInstance(example_path)
        print("Running GA...")
        res = solve_ga(inst, time_limit=30)
        print("Result:", res)
