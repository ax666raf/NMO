"""
Experiments & Plots — Drone-Based Delivery Optimization
========================================================
Runs all methods, compiles results, and produces all figures for the report.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from instance_generator import INSTANCE_CONFIGS, load_instance
from branch_and_bound import BranchAndBound
from genetic_algorithm import genetic_algorithm
from simulated_annealing import simulated_annealing

BASE = os.path.join(os.path.dirname(__file__), "..")
INST_DIR = os.path.join(BASE, "instances")
RES_DIR  = os.path.join(BASE, "results")
PLT_DIR  = os.path.join(BASE, "results", "plots")
os.makedirs(PLT_DIR, exist_ok=True)

STYLE = {
    "BranchAndBound":     {"color": "#2E86C1", "marker": "o", "ls": "-",  "label": "Branch & Bound"},
    "GeneticAlgorithm":   {"color": "#E74C3C", "marker": "s", "ls": "--", "label": "Genetic Algorithm"},
    "SimulatedAnnealing": {"color": "#27AE60", "marker": "^", "ls": ":",  "label": "Simulated Annealing"},
}


# ── 1. Run all methods ────────────────────────────────────────────────────────

def run_all():
    bb_results, ga_results, sa_results = [], [], []

    for (iid, n, k, seed) in INSTANCE_CONFIGS:
        path = os.path.join(INST_DIR, f"{iid}.json")
        inst = load_instance(path)

        # B&B only on small instances
        if n <= 10:
            print(f"[B&B] {iid} (n={n}, k={k})", end=" ... ", flush=True)
            bb = BranchAndBound(inst, time_limit=60)
            r = bb.solve()
            print(f"cost={r['best_cost']:.2f}, nodes={r['nodes_explored']}, "
                  f"{'OPTIMAL' if r['optimal'] else 'TIMEOUT'}")
            bb_results.append({
                "instance": iid, "n": n, "k": k,
                "method": "BranchAndBound",
                "cost": r["best_cost"], "time": r["elapsed"],
                "nodes": r["nodes_explored"], "optimal": r["optimal"],
            })

        # GA — all instances
        print(f"[GA]  {iid} (n={n}, k={k})", end=" ... ", flush=True)
        r = genetic_algorithm(inst, pop_size=50, n_gen=200, seed=seed)
        print(f"cost={r['best_cost']:.2f}, time={r['elapsed']:.2f}s")
        ga_results.append({
            "instance": iid, "n": n, "k": k,
            "method": "GeneticAlgorithm",
            "cost": r["best_cost"], "time": r["elapsed"],
            "cost_history": r["cost_history"],
        })

        # SA — all instances
        print(f"[SA]  {iid} (n={n}, k={k})", end=" ... ", flush=True)
        r = simulated_annealing(inst, seed=seed)
        print(f"cost={r['best_cost']:.2f}, time={r['elapsed']:.2f}s")
        sa_results.append({
            "instance": iid, "n": n, "k": k,
            "method": "SimulatedAnnealing",
            "cost": r["best_cost"], "time": r["elapsed"],
            "cost_history": r["cost_history"],
        })

    # Save raw results
    for name, data in [("bb", bb_results), ("ga", ga_results), ("sa", sa_results)]:
        with open(os.path.join(RES_DIR, f"{name}_results.json"), "w") as f:
            json.dump(data, f, indent=2)

    return bb_results, ga_results, sa_results


# ── 2. Summary table ──────────────────────────────────────────────────────────

def print_table(bb_results, ga_results, sa_results):
    bb_map = {r["instance"]: r for r in bb_results}
    ga_map = {r["instance"]: r for r in ga_results}
    sa_map = {r["instance"]: r for r in sa_results}

    header = f"{'Inst':>4} {'n':>3} {'k':>2} | {'B&B Cost':>10} {'B&B Time':>9} {'Optimal':>7} | {'GA Cost':>9} {'GA Time':>8} | {'SA Cost':>9} {'SA Time':>8} | {'GA Gap%':>8} {'SA Gap%':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for (iid, n, k, seed) in INSTANCE_CONFIGS:
        ga = ga_map.get(iid)
        sa = sa_map.get(iid)
        bb = bb_map.get(iid)

        bb_str  = f"{bb['cost']:10.2f} {bb['time']:9.3f}s {'YES' if bb['optimal'] else 'NO':>7}" if bb else f"{'N/A':>10} {'N/A':>9} {'N/A':>7}"
        ga_str  = f"{ga['cost']:9.2f} {ga['time']:8.3f}s" if ga else f"{'N/A':>9} {'N/A':>8}"
        sa_str  = f"{sa['cost']:9.2f} {sa['time']:8.3f}s" if sa else f"{'N/A':>9} {'N/A':>8}"

        if bb and ga:
            ga_gap = 100 * (ga["cost"] - bb["cost"]) / bb["cost"]
            ga_gap_str = f"{ga_gap:8.2f}%"
        else:
            ga_gap_str = f"{'N/A':>8}"

        if bb and sa:
            sa_gap = 100 * (sa["cost"] - bb["cost"]) / bb["cost"]
            sa_gap_str = f"{sa_gap:8.2f}%"
        else:
            sa_gap_str = f"{'N/A':>8}"

        print(f"{iid:>4} {n:>3} {k:>2} | {bb_str} | {ga_str} | {sa_str} | {ga_gap_str} {sa_gap_str}")

    print("=" * len(header) + "\n")


# ── 3. Figure 1: Cost vs Instance Size ───────────────────────────────────────

def plot_cost_vs_size(bb_results, ga_results, sa_results):
    fig, ax = plt.subplots(figsize=(9, 5))

    for data, key in [(ga_results, "GeneticAlgorithm"), (sa_results, "SimulatedAnnealing")]:
        ns = [r["n"] for r in data]
        costs = [r["cost"] for r in data]
        s = STYLE[key]
        ax.plot(ns, costs, color=s["color"], marker=s["marker"],
                ls=s["ls"], label=s["label"], linewidth=2, markersize=7)

    # B&B only on small
    if bb_results:
        ns = [r["n"] for r in bb_results]
        costs = [r["cost"] for r in bb_results]
        s = STYLE["BranchAndBound"]
        ax.plot(ns, costs, color=s["color"], marker=s["marker"],
                ls=s["ls"], label=s["label"], linewidth=2, markersize=7)

    ax.set_xlabel("Number of Customers (n)", fontsize=12)
    ax.set_ylabel("Total Energy Cost", fontsize=12)
    ax.set_title("Solution Quality vs Instance Size", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLT_DIR, "fig1_cost_vs_size.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── 4. Figure 2: Runtime vs Instance Size ────────────────────────────────────

def plot_runtime_vs_size(bb_results, ga_results, sa_results):
    fig, ax = plt.subplots(figsize=(9, 5))

    for data, key in [(ga_results, "GeneticAlgorithm"), (sa_results, "SimulatedAnnealing")]:
        ns = [r["n"] for r in data]
        times = [r["time"] for r in data]
        s = STYLE[key]
        ax.plot(ns, times, color=s["color"], marker=s["marker"],
                ls=s["ls"], label=s["label"], linewidth=2, markersize=7)

    if bb_results:
        ns = [r["n"] for r in bb_results]
        times = [r["time"] for r in bb_results]
        s = STYLE["BranchAndBound"]
        ax.plot(ns, times, color=s["color"], marker=s["marker"],
                ls=s["ls"], label=s["label"], linewidth=2, markersize=7)

    ax.set_xlabel("Number of Customers (n)", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("Runtime vs Instance Size", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLT_DIR, "fig2_runtime_vs_size.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── 5. Figure 3: GA Convergence Curve ────────────────────────────────────────

def plot_ga_convergence(ga_results):
    # Use instance I07 (n=15) as representative
    target = next((r for r in ga_results if r["instance"] == "I07"), ga_results[-1])
    history = target["cost_history"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(len(history)), history, color=STYLE["GeneticAlgorithm"]["color"],
            linewidth=2, label=f"Instance {target['instance']} (n={target['n']}, k={target['k']})")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness (Energy Cost)", fontsize=12)
    ax.set_title("GA Convergence Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLT_DIR, "fig3_ga_convergence.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── 6. Figure 4: SA Cooling Curve ────────────────────────────────────────────

def plot_sa_cooling(sa_results):
    target = next((r for r in sa_results if r["instance"] == "I07"), sa_results[-1])
    history = target["cost_history"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(len(history)), history, color=STYLE["SimulatedAnnealing"]["color"],
            linewidth=1.5, alpha=0.85,
            label=f"Instance {target['instance']} (n={target['n']}, k={target['k']})")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Cost Found", fontsize=12)
    ax.set_title("SA Cooling Curve (Best Cost over Iterations)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLT_DIR, "fig4_sa_cooling.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── 7. Figure 5: Gap comparison bar chart (small instances only) ─────────────

def plot_gap_bars(bb_results, ga_results, sa_results):
    bb_map = {r["instance"]: r["cost"] for r in bb_results}
    ga_map = {r["instance"]: r["cost"] for r in ga_results}
    sa_map = {r["instance"]: r["cost"] for r in sa_results}

    instances = sorted(bb_map.keys())
    ga_gaps = [100 * (ga_map[i] - bb_map[i]) / bb_map[i] for i in instances]
    sa_gaps = [100 * (sa_map[i] - bb_map[i]) / bb_map[i] for i in instances]

    x = np.arange(len(instances))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, ga_gaps, width, label="GA Gap %",
           color=STYLE["GeneticAlgorithm"]["color"], alpha=0.85)
    ax.bar(x + width/2, sa_gaps, width, label="SA Gap %",
           color=STYLE["SimulatedAnnealing"]["color"], alpha=0.85)

    ax.set_xlabel("Instance", fontsize=12)
    ax.set_ylabel("Gap from B&B Optimal (%)", fontsize=12)
    ax.set_title("Metaheuristic Gap from Optimal (small instances)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(instances)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    out = os.path.join(PLT_DIR, "fig5_gap_bars.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Running all experiments...")
    print("=" * 60)

    bb_results, ga_results, sa_results = run_all()

    print("\n--- RESULTS TABLE ---")
    print_table(bb_results, ga_results, sa_results)

    print("--- GENERATING PLOTS ---")
    plot_cost_vs_size(bb_results, ga_results, sa_results)
    plot_runtime_vs_size(bb_results, ga_results, sa_results)
    plot_ga_convergence(ga_results)
    plot_sa_cooling(sa_results)
    plot_gap_bars(bb_results, ga_results, sa_results)

    print("\nAll done. Check results/plots/ for figures.")
