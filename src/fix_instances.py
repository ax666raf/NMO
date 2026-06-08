"""Regenerate instances with scaled battery limits based on instance size."""
from instance_generator import generate_instance, save_instance, INSTANCE_CONFIGS
import os

# Battery scales with n: larger instances need proportionally more battery
# Rule: B = 150 + 15*n (gives room for ~n/k customers per route + return)
configs_scaled = [
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

out_dir = "../instances"
os.makedirs(out_dir, exist_ok=True)

for (iid, n, k, seed) in configs_scaled:
    battery = 150 + 15 * n   # scales sensibly
    payload = 8 * (n // k) + 5  # per-drone payload ~ avg demand per drone + buffer
    inst = generate_instance(n, k, payload_capacity=payload, battery_limit=battery, seed=seed)
    path = os.path.join(out_dir, f"{iid}.json")
    save_instance(inst, path)
    print(f"{iid}: n={n}, k={k}, Q={payload}, B={battery}")

print("Done.")
