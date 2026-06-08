"""
Simulated Annealing — Drone-Based Delivery Optimization
========================================================
Representation : list of k routes (lists of customer node indices)
Initial solution: greedy nearest-neighbour
Neighbourhood  : Relocate (move customer between drones) +
                 2-opt within a single drone route
Cooling        : Geometric  T = T * alpha  until T < T_min
"""

import math
import random
import time
import copy
import json
import os
from instance_generator import load_instance, INSTANCE_CONFIGS


# ── Solution helpers ─────────────────────────────────────────────────────────

def route_cost(route, dist):
    if not route:
        return 0.0
    cost = dist[0][route[0]]
    for i in range(len(route) - 1):
        cost += dist[route[i]][route[i + 1]]
    cost += dist[route[-1]][0]
    return cost


def solution_cost(routes, dist):
    return sum(route_cost(r, dist) for r in routes)


def is_feasible(routes, demands, Q, B, dist):
    for route in routes:
        if not route:
            continue
        load = sum(demands[c - 1] for c in route)
        if load > Q:
            return False
        if route_cost(route, dist) > B:
            return False
    return True


# ── Greedy initial solution ───────────────────────────────────────────────────

def greedy_init(inst):
    n = inst["n_customers"]
    k = inst["n_drones"]
    dist = inst["dist"]
    demands = inst["demands"]
    Q = inst["payload_capacity"]
    B = inst["battery_limit"]

    routes = [[] for _ in range(k)]
    loads = [0.0] * k
    energies = [0.0] * k
    positions = [0] * k
    unvisited = list(range(1, n + 1))

    while unvisited:
        best = (None, None, math.inf)
        for d in range(k):
            for c in unvisited:
                if loads[d] + demands[c - 1] > Q:
                    continue
                go = dist[positions[d]][c]
                ret = dist[c][0]
                if energies[d] + go + ret > B:
                    continue
                if go < best[2]:
                    best = (d, c, go)

        if best[0] is None:
            # Assign remaining to least-loaded drone regardless
            c = unvisited[0]
            d = min(range(k), key=lambda x: loads[x])
        else:
            d, c, _ = best

        routes[d].append(c)
        loads[d] += demands[c - 1]
        energies[d] += dist[positions[d]][c]
        positions[d] = c
        unvisited.remove(c)

    return routes


# ── Neighbourhood operators ───────────────────────────────────────────────────

def relocate(routes, demands, Q, B, dist):
    """
    Move one random customer from one drone to another.
    Returns new routes if feasible improvement found, else None.
    """
    k = len(routes)
    non_empty = [d for d in range(k) if routes[d]]
    if len(non_empty) < 1:
        return None

    d_from = random.choice(non_empty)
    if not routes[d_from]:
        return None
    pos = random.randrange(len(routes[d_from]))
    customer = routes[d_from][pos]

    # Try inserting into a different drone at a random position
    candidates = [d for d in range(k) if d != d_from]
    if not candidates:
        return None
    d_to = random.choice(candidates)
    ins_pos = random.randint(0, len(routes[d_to]))

    new_routes = copy.deepcopy(routes)
    new_routes[d_from].pop(pos)
    new_routes[d_to].insert(ins_pos, customer)

    # Feasibility check
    load_to = sum(demands[c - 1] for c in new_routes[d_to])
    if load_to > Q:
        return None
    if route_cost(new_routes[d_to], dist) > B:
        return None

    return new_routes


def two_opt(routes, dist):
    """
    Apply 2-opt improvement within a random drone's route.
    Always returns a (possibly improved) copy.
    """
    k = len(routes)
    non_empty = [d for d in range(k) if len(routes[d]) >= 2]
    if not non_empty:
        return None

    d = random.choice(non_empty)
    route = routes[d][:]
    n = len(route)
    i, j = sorted(random.sample(range(n), 2))

    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
    new_routes = copy.deepcopy(routes)
    new_routes[d] = new_route
    return new_routes


def swap_between_drones(routes, demands, Q, B, dist):
    """Swap one customer from drone A with one from drone B."""
    k = len(routes)
    non_empty = [d for d in range(k) if routes[d]]
    if len(non_empty) < 2:
        return None

    d1, d2 = random.sample(non_empty, 2)
    if not routes[d1] or not routes[d2]:
        return None

    p1 = random.randrange(len(routes[d1]))
    p2 = random.randrange(len(routes[d2]))
    c1 = routes[d1][p1]
    c2 = routes[d2][p2]

    new_routes = copy.deepcopy(routes)
    new_routes[d1][p1] = c2
    new_routes[d2][p2] = c1

    # Feasibility
    for d in [d1, d2]:
        load = sum(demands[c - 1] for c in new_routes[d])
        if load > Q:
            return None
        if route_cost(new_routes[d], dist) > B:
            return None

    return new_routes


def get_neighbour(routes, demands, Q, B, dist):
    """Randomly apply one of the three neighbourhood operators."""
    op = random.randint(0, 2)
    if op == 0:
        return relocate(routes, demands, Q, B, dist)
    elif op == 1:
        return two_opt(routes, dist)
    else:
        return swap_between_drones(routes, demands, Q, B, dist)


# ── Simulated Annealing ───────────────────────────────────────────────────────

def simulated_annealing(inst, T_init=100.0, T_min=0.01, alpha=0.995,
                        max_iter=5000, seed=0, verbose=False):
    random.seed(seed)

    dist = inst["dist"]
    demands = inst["demands"]
    Q = inst["payload_capacity"]
    B = inst["battery_limit"]

    # Initial solution
    current = greedy_init(inst)
    current_cost = solution_cost(current, dist)

    best = copy.deepcopy(current)
    best_cost = current_cost

    T = T_init
    cost_history = [current_cost]
    start = time.time()

    for iteration in range(max_iter):
        # Generate neighbour
        neighbour = None
        attempts = 0
        while neighbour is None and attempts < 20:
            neighbour = get_neighbour(current, demands, Q, B, dist)
            attempts += 1

        if neighbour is None:
            T *= alpha
            cost_history.append(current_cost)
            continue

        neighbour_cost = solution_cost(neighbour, dist)
        delta = neighbour_cost - current_cost

        # Accept or reject
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbour
            current_cost = neighbour_cost

        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best = copy.deepcopy(current)

        cost_history.append(best_cost)
        T *= alpha

        if T < T_min:
            break

        if verbose and iteration % 1000 == 0:
            print(f"  iter={iteration}, T={T:.4f}, best={best_cost:.2f}")

    elapsed = time.time() - start
    return {
        "best_cost": best_cost,
        "best_routes": best,
        "cost_history": cost_history,
        "elapsed": elapsed,
        "iterations": len(cost_history),
    }


# ── Run on all instances ──────────────────────────────────────────────────────

def run_all(instances_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    results = []

    for (iid, n, k, seed) in INSTANCE_CONFIGS:
        path = os.path.join(instances_dir, f"{iid}.json")
        inst = load_instance(path)

        print(f"Running SA on {iid} (n={n}, k={k}) ... ", end="", flush=True)
        res = simulated_annealing(inst, seed=seed, verbose=False)
        print(f"cost={res['best_cost']:.2f}, time={res['elapsed']:.2f}s, "
              f"iters={res['iterations']}")

        results.append({
            "instance": iid,
            "n": n,
            "k": k,
            "method": "SimulatedAnnealing",
            "cost": res["best_cost"],
            "time": res["elapsed"],
            "cost_history": res["cost_history"],
        })

    out_path = os.path.join(results_dir, "sa_results.json")
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
