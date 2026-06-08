"""
Instance Generator — Drone-Based Delivery Optimization
Generates reproducible random instances for all methods.
"""

import numpy as np
import json
import os

def generate_instance(n_customers, n_drones, payload_capacity=25, battery_limit=200, seed=42):
    """
    Generate one problem instance.

    Parameters
    ----------
    n_customers     : number of customers to serve
    n_drones        : number of drones available
    payload_capacity: max weight each drone can carry
    battery_limit   : max total distance each drone can travel
    seed            : random seed for reproducibility

    Returns
    -------
    dict with all instance data
    """
    rng = np.random.default_rng(seed)

    # Depot at center
    depot = np.array([50.0, 50.0])

    # Customer coordinates in 100x100 grid
    customers = rng.uniform(0, 100, size=(n_customers, 2))

    # Customer demands (weight): integer in [1, 10]
    demands = rng.integers(1, 11, size=n_customers)

    # Build full coordinate array: index 0 = depot, 1..n = customers
    coords = np.vstack([depot, customers])

    # Distance matrix (Euclidean) — also serves as energy cost
    n_nodes = n_customers + 1
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist[i][j] = np.linalg.norm(coords[i] - coords[j])

    return {
        "n_customers": n_customers,
        "n_drones": n_drones,
        "payload_capacity": payload_capacity,
        "battery_limit": battery_limit,
        "seed": seed,
        "depot": depot.tolist(),
        "customers": customers.tolist(),
        "demands": demands.tolist(),
        "coords": coords.tolist(),
        "dist": dist.tolist(),
    }


def save_instance(instance, path):
    with open(path, "w") as f:
        json.dump(instance, f, indent=2)


def load_instance(path):
    with open(path) as f:
        inst = json.load(f)
    inst["dist"] = np.array(inst["dist"])
    inst["demands"] = np.array(inst["demands"])
    inst["coords"] = np.array(inst["coords"])
    return inst


# ── Instance suite ──────────────────────────────────────────────────────────
INSTANCE_CONFIGS = [
    # (id, n_customers, n_drones, seed)   — 10 instances total
    ("I01",  5, 2,  1),
    ("I02",  5, 2,  2),
    ("I03",  8, 2,  3),
    ("I04", 10, 3,  4),
    ("I05", 10, 3,  5),
    ("I06", 12, 3,  6),
    ("I07", 15, 4,  7),
    ("I08", 15, 4,  8),
    ("I09", 20, 4,  9),
    ("I10", 25, 5, 10),
]


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "instances")
    os.makedirs(out_dir, exist_ok=True)

    for (iid, n, k, seed) in INSTANCE_CONFIGS:
        inst = generate_instance(n, k, seed=seed)
        path = os.path.join(out_dir, f"{iid}.json")
        save_instance(inst, path)
        print(f"Generated {iid}: n={n}, k={k}, seed={seed}  →  {path}")

    print("\nAll instances saved.")
