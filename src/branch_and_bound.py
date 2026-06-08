"""
Branch and Bound — Drone-Based Delivery Optimization
=====================================================
Exact method. Practical only for small instances (n <= 10).
Uses DFS with:
  - Greedy initial upper bound (nearest-neighbour heuristic)
  - Lower bound = current cost + MST cost of unvisited customers
  - Feasibility pruning on payload and battery constraints
"""

import time
import math
import itertools
import numpy as np
from instance_generator import load_instance, INSTANCE_CONFIGS
import json, os

# ── Greedy initial solution (nearest-neighbour) ──────────────────────────────

def greedy_solution(inst):
    """
    Build a feasible solution greedily:
    For each unassigned customer, assign to the drone that can reach it
    with the least added cost and still return to depot.
    """
    n = inst["n_customers"]
    k = inst["n_drones"]
    dist = inst["dist"]
    demands = inst["demands"]
    Q = inst["payload_capacity"]
    B = inst["battery_limit"]

    routes = [[] for _ in range(k)]
    loads = [0.0] * k      # current payload per drone
    energies = [0.0] * k   # current energy used per drone
    positions = [0] * k    # current node (0 = depot)

    unvisited = list(range(1, n + 1))  # node indices (1-based, 0=depot)

    while unvisited:
        best_drone, best_customer, best_delta = -1, -1, math.inf

        for d in range(k):
            for c in unvisited:
                demand = demands[c - 1]
                if loads[d] + demand > Q:
                    continue
                go = dist[positions[d]][c]
                ret = dist[c][0]
                if energies[d] + go + ret > B:
                    continue
                delta = go
                if delta < best_delta:
                    best_delta = delta
                    best_drone = d
                    best_customer = c

        if best_drone == -1:
            # No feasible assignment — force into least-loaded drone ignoring battery
            # (signals infeasibility; B&B will handle it)
            break

        d, c = best_drone, best_customer
        energies[d] += dist[positions[d]][c]
        positions[d] = c
        loads[d] += demands[c - 1]
        routes[d].append(c)
        unvisited.remove(c)

    # Add return-to-depot cost
    total_cost = sum(energies[d] + dist[positions[d]][0] for d in range(k))
    return routes, total_cost


# ── Route cost helper ────────────────────────────────────────────────────────

def route_cost(route, dist):
    """Total distance: depot -> route -> depot."""
    if not route:
        return 0.0
    cost = dist[0][route[0]]
    for i in range(len(route) - 1):
        cost += dist[route[i]][route[i + 1]]
    cost += dist[route[-1]][0]
    return cost


def solution_cost(routes, dist):
    return sum(route_cost(r, dist) for r in routes)


# ── Minimum Spanning Tree (Prim) for lower bound ─────────────────────────────

def mst_cost(nodes, dist):
    """
    Compute MST cost over a set of nodes (including depot=0) using Prim's.
    Used as a lower bound on the remaining delivery cost.
    """
    if len(nodes) <= 1:
        return 0.0
    nodes = list(nodes)
    in_tree = {nodes[0]}
    edges = []
    total = 0.0
    while len(in_tree) < len(nodes):
        best = math.inf
        best_node = -1
        for u in in_tree:
            for v in nodes:
                if v not in in_tree:
                    if dist[u][v] < best:
                        best = dist[u][v]
                        best_node = v
        in_tree.add(best_node)
        total += best
    return total


# ── Branch and Bound ─────────────────────────────────────────────────────────

class BranchAndBound:
    def __init__(self, inst, time_limit=60):
        self.inst = inst
        self.n = inst["n_customers"]
        self.k = inst["n_drones"]
        self.dist = inst["dist"]
        self.demands = inst["demands"]
        self.Q = inst["payload_capacity"]
        self.B = inst["battery_limit"]
        self.time_limit = time_limit

        self.best_cost = math.inf
        self.best_routes = None
        self.nodes_explored = 0
        self.start_time = None
        self.timed_out = False

    def lower_bound(self, routes, positions, loads, energies, unvisited):
        """
        LB = current energy already spent
             + for each drone: distance from current position to nearest unvisited or depot
             + MST over unvisited customers + depot
        """
        current = sum(energies)

        # Each drone must return to depot eventually
        drone_return = sum(self.dist[positions[d]][0] for d in range(self.k))

        # MST over unvisited + depot gives optimistic remaining delivery cost
        mst_nodes = {0} | set(unvisited)
        mst = mst_cost(mst_nodes, self.dist)

        return current + max(drone_return, mst)

    def solve(self):
        self.start_time = time.time()

        # Warm start with greedy solution
        greedy_routes, greedy_cost = greedy_solution(self.inst)
        self.best_cost = greedy_cost
        self.best_routes = greedy_routes

        unvisited = set(range(1, self.n + 1))
        routes = [[] for _ in range(self.k)]
        positions = [0] * self.k
        loads = [0.0] * self.k
        energies = [0.0] * self.k

        self._dfs(routes, positions, loads, energies, unvisited)

        elapsed = time.time() - self.start_time
        return {
            "best_cost": self.best_cost,
            "best_routes": self.best_routes,
            "nodes_explored": self.nodes_explored,
            "elapsed": elapsed,
            "timed_out": self.timed_out,
            "optimal": not self.timed_out,
        }

    def _dfs(self, routes, positions, loads, energies, unvisited):
        # Time limit check
        if time.time() - self.start_time > self.time_limit:
            self.timed_out = True
            return

        self.nodes_explored += 1

        # All customers assigned → evaluate complete solution
        if not unvisited:
            total = sum(
                energies[d] + self.dist[positions[d]][0]
                for d in range(self.k)
            )
            if total < self.best_cost:
                self.best_cost = total
                self.best_routes = [list(r) for r in routes]
            return

        # Lower bound pruning
        lb = self.lower_bound(routes, positions, loads, energies, unvisited)
        if lb >= self.best_cost:
            return

        # Branch: pick next customer to assign (smallest index for symmetry breaking)
        customer = min(unvisited)
        demand = self.demands[customer - 1]

        # Try assigning to each drone (skip symmetric empty drones)
        tried_empty = False
        for d in range(self.k):
            # Symmetry breaking: don't try multiple empty drones
            if not routes[d]:
                if tried_empty:
                    continue
                tried_empty = True

            # Feasibility checks
            if loads[d] + demand > self.Q:
                continue
            go = self.dist[positions[d]][customer]
            ret = self.dist[customer][0]
            if energies[d] + go + ret > self.B:
                continue

            # Branch: assign customer to drone d
            routes[d].append(customer)
            old_pos = positions[d]
            old_load = loads[d]
            old_energy = energies[d]

            positions[d] = customer
            loads[d] += demand
            energies[d] += go
            unvisited.remove(customer)

            self._dfs(routes, positions, loads, energies, unvisited)

            if self.timed_out:
                return

            # Backtrack
            routes[d].pop()
            positions[d] = old_pos
            loads[d] = old_load
            energies[d] = old_energy
            unvisited.add(customer)


# ── Run on all instances ─────────────────────────────────────────────────────

def run_all(instances_dir, results_dir, time_limit=60):
    os.makedirs(results_dir, exist_ok=True)
    results = []

    # Only run B&B on small instances (n <= 10)
    small = [cfg for cfg in INSTANCE_CONFIGS if cfg[1] <= 10]

    for (iid, n, k, seed) in small:
        path = os.path.join(instances_dir, f"{iid}.json")
        inst = load_instance(path)

        print(f"Running B&B on {iid} (n={n}, k={k}) ... ", end="", flush=True)
        bb = BranchAndBound(inst, time_limit=time_limit)
        res = bb.solve()

        status = "OPTIMAL" if res["optimal"] else f"TIMEOUT ({time_limit}s)"
        print(f"cost={res['best_cost']:.2f}, nodes={res['nodes_explored']}, "
              f"time={res['elapsed']:.2f}s, {status}")

        results.append({
            "instance": iid,
            "n": n,
            "k": k,
            "method": "BranchAndBound",
            "cost": res["best_cost"],
            "time": res["elapsed"],
            "nodes_explored": res["nodes_explored"],
            "optimal": res["optimal"],
        })

    # Save
    out_path = os.path.join(results_dir, "bb_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..")
    run_all(
        instances_dir=os.path.join(base, "instances"),
        results_dir=os.path.join(base, "results"),
        time_limit=60,
    )
