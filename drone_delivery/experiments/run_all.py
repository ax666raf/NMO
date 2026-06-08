import sys
import os
import glob
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'solvers')))
from utils import DroneInstance
from branch_and_bound import solve_bnb
from genetic_algorithm import solve_ga
from tabu_search import solve_tabu

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.abspath(os.path.join(base_dir, '..', 'data', 'instances'))
    results_dir = os.path.join(base_dir, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    instance_files = []
    for cat in ['small', 'medium', 'large']:
        cat_dir = os.path.join(instances_dir, cat)
        instance_files.extend(glob.glob(os.path.join(cat_dir, '*.json')))
        
    instance_files.sort()
    results = []
    limits = {'BnB': 30, 'GA': 5, 'TS': 5}
    print("Running experiments... ")
    
    for fpath in tqdm(instance_files):
        inst = DroneInstance(fpath)
        
        try:
            res_bnb = solve_bnb(inst, time_limit=limits['BnB'])
        except Exception as e:
            res_bnb = {'cost': float('inf'), 'runtime': limits['BnB'], 'gap': 0, 'feasible': False}
            
        results.append({'instance_id': inst.instance_id, 'n_customers': inst.n_customers, 'method': 'BnB', 'cost': res_bnb['cost'], 'runtime_sec': res_bnb['runtime'], 'gap_pct': res_bnb['gap'], 'feasible': res_bnb['feasible']})
        
        try:
            res_ga = solve_ga(inst, time_limit=limits['GA'])
        except Exception as e:
            res_ga = {'cost': float('inf'), 'runtime': limits['GA'], 'gap': '-', 'feasible': False}
            
        results.append({'instance_id': inst.instance_id, 'n_customers': inst.n_customers, 'method': 'GA', 'cost': res_ga['cost'], 'runtime_sec': res_ga['runtime'], 'gap_pct': res_ga['gap'], 'feasible': res_ga['feasible']})
        
        try:
            res_ts = solve_tabu(inst, time_limit=limits['TS'])
        except Exception as e:
            res_ts = {'cost': float('inf'), 'runtime': limits['TS'], 'gap': '-', 'feasible': False}
            
        results.append({'instance_id': inst.instance_id, 'n_customers': inst.n_customers, 'method': 'TS', 'cost': res_ts['cost'], 'runtime_sec': res_ts['runtime'], 'gap_pct': res_ts['gap'], 'feasible': res_ts['feasible']})
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, 'all_results_preview.csv'), index=False)
    
    print("\nExperiments completed. Summary by method:")
    valid_df = df[df['feasible'] == True]
    summary = valid_df.groupby('method').agg({'cost': 'mean', 'runtime_sec': 'mean'}).round(2)
    print(summary)
    
if __name__ == '__main__':
    main()
