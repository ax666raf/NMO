"""
Genetic Algorithm — Drone-Based Delivery Optimization
======================================================
Encoding  : Giant permutation of customers, split into k drone routes
Crossover : Order Crossover (OX)
Mutation  : Random swap of two customers in the permutation
Selection : Tournament selection
Repair    : Move overloading/over-battery customer to least-loaded drone
"""

import numpy as np
import random
import time
import json
import os
from instance_generator import load_instance, INSTANCE_CONFIGS

# ── Decoder: permutation → routes ───────────────────────────────────────────

def decode(permutation, n_drones, demands, Q, B, dist):
    """
    Split a permutation of customers into k drone routes.
    For each customer, try every drone in order and assign to the first feasible one.
    If no drone is feasible, assign to the drone with the most remaining capacity
    (penalty will handle the violation in fitness).
    """
    routes = [[] for _ in range(n_drones)]
    loads = [0.0] * n_drones
    energies = [0.0] * n_drones
    positions = [0] * n_drones  # all start at depot

    for customer in permutation:
        demand = demands[customer - 1]
        assigned = False

        # Try each drone in order — pick first feasible
        for d in range(n_drones):
            go = dist[positions[d]][customer]
            ret = dist[customer][0]
            if (loads[d] + demand <= Q and
                    energies[d] + go + ret <= B):
                routes[d].append(customer)
                loads[d] += demand
                energies[d] += go
                positions[d] = customer
                assigned = True
                break

        if not assigned:
            # Force to least-loaded drone (penalty will fire)
            d = min(range(n_drones), key=lambda x: loads[x])
            routes[d].append(customer)
            loads[d] += demand
            energies[d] += dist[positions[d]][customer]
            positions[d] = customer

    return routes


def fitness(permutation, inst):
    """
    Compute total energy cost. Add penalty for constraint violations.
    """
    n = inst["n_customers"]
    k = inst["n_drones"]
    dist = inst["dist"]
    demands = inst["demands"]
    Q = inst["payload_capacity"]
    B = inst["battery_limit"]

    routes = decode(permutation, k, demands, Q, B, dist)

    total_cost = 0.0
    penalty = 0.0

    for route in routes:
        if not route:
            continue
        # Route cost
        cost = dist[0][route[0]]
        for i in range(len(route) - 1):
            cost += dist[route[i]][route[i + 1]]
        cost += dist[route[-1]][0]
        total_cost += cost

        # Payload violation
        load = sum(demands[c - 1] for c in route)
        if load > Q:
            penalty += 1000 * (load - Q)

        # Battery violation
        if cost > B:
            penalty += 1000 * (cost - B)

    return total_cost + penalty, routes


# ── Genetic operators ────────────────────────────────────────────────────────

def order_crossover(p1, p2):
    """OX crossover: preserves relative order from one parent."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b+1] = p1[a:b+1]
    segment = set(p1[a:b+1])
    ptr = 0
    for gene in p2:
        if gene not in segment:
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = gene
    return child


def mutate(permutation, mutation_rate=0.1):
    """Swap mutation: randomly swap two positions."""
    perm = permutation[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def tournament_select(population, fitnesses, k=3):
    """Select best individual among k random candidates."""
    candidates = random.sample(range(len(population)), k)
    best = min(candidates, key=lambda i: fitnesses[i])
    return population[best][:]


# ── Main GA loop ─────────────────────────────────────────────────────────────

def genetic_algorithm(inst, pop_size=50, n_gen=200, cx_prob=0.8,
                      mut_rate=0.1, elitism=2, seed=0, verbose=False):
    random.seed(seed)
    np.random.seed(seed)

    n = inst["n_customers"]
    customers = list(range(1, n + 1))

    # Initial population: random permutations
    population = [random.sample(customers, n) for _ in range(pop_size)]

    best_cost_history = []
    best_solution = None
    best_cost = float("inf")
    best_routes = None

    start = time.time()

    for gen in range(n_gen):
        # Evaluate
        evals = [fitness(ind, inst) for ind in population]
        costs = [e[0] for e in evals]
        routes_list = [e[1] for e in evals]

        # Track best
        gen_best_idx = min(range(pop_size), key=lambda i: costs[i])
        if costs[gen_best_idx] < best_cost:
            best_cost = costs[gen_best_idx]
            best_solution = population[gen_best_idx][:]
            best_routes = routes_list[gen_best_idx]

        best_cost_history.append(best_cost)

        if verbose and gen % 50 == 0:
            print(f"  Gen {gen:4d}: best_cost={best_cost:.2f}")

        # Elitism: carry top individuals to next generation
        sorted_idx = sorted(range(pop_size), key=lambda i: costs[i])
        new_population = [population[i][:] for i in sorted_idx[:elitism]]

        # Fill rest with crossover + mutation
        while len(new_population) < pop_size:
            p1 = tournament_select(population, costs)
            p2 = tournament_select(population, costs)
            if random.random() < cx_prob:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]
            child = mutate(child, mut_rate)
            new_population.append(child)

        population = new_population

    elapsed = time.time() - start
    return {
        "best_cost": best_cost,
        "best_routes": best_routes,
        "cost_history": best_cost_history,
        "elapsed": elapsed,
    }


# ── Run on all instances ─────────────────────────────────────────────────────

def run_all(instances_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    results = []

    for (iid, n, k, seed) in INSTANCE_CONFIGS:
        path = os.path.join(instances_dir, f"{iid}.json")
        inst = load_instance(path)

        print(f"Running GA on {iid} (n={n}, k={k}) ... ", end="", flush=True)
        res = genetic_algorithm(inst, pop_size=50, n_gen=200, seed=seed, verbose=False)
        print(f"cost={res['best_cost']:.2f}, time={res['elapsed']:.2f}s")

        results.append({
            "instance": iid,
            "n": n,
            "k": k,
            "method": "GeneticAlgorithm",
            "cost": res["best_cost"],
            "time": res["elapsed"],
            "cost_history": res["cost_history"],
        })

    out_path = os.path.join(results_dir, "ga_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..")
    run_all(
        instances_dir=os.path.join(base, "instances"),
        results_dir=os.path.join(base, "results"),
    )
