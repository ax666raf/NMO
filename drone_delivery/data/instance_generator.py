import json
import os
import random

def generate_instance(instance_id, n_customers, n_drones, battery_km, payload_kg):
    depot = [0.0, 0.0]
    
    # Generate random customer coordinates in [0, 100] x [0, 100]
    customers = [[round(random.uniform(0, 100), 2), round(random.uniform(0, 100), 2)]
                 for _ in range(n_customers)]
                 
    # Generate random demands in [0.5, 4.0]
    demands = [round(random.uniform(0.5, 4.0), 2) for _ in range(n_customers)]
    
    return {
        "instance_id": instance_id,
        "n_customers": n_customers,
        "n_drones": n_drones,
        "depot": depot,
        "customers": customers,
        "demands": demands,
        "battery_km": battery_km,
        "payload_kg": payload_kg,
        "no_fly_zones": []
    }

def main():
    # Configure random seed for reproducibility
    random.seed(42)
    
    categories = {
        "small": {"n_range": (5, 8), "k": 2, "battery": 80.0, "payload": 5.0, "count": 4},
        "medium": {"n_range": (10, 15), "k": 3, "battery": 100.0, "payload": 7.0, "count": 4},
        "large": {"n_range": (20, 30), "k": 4, "battery": 120.0, "payload": 10.0, "count": 4}
    }
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for category, params in categories.items():
        out_dir = os.path.join(base_dir, "instances", category)
        os.makedirs(out_dir, exist_ok=True)
        
        for i in range(1, params["count"] + 1):
            n_customers = random.randint(params["n_range"][0], params["n_range"][1])
            instance_id = f"{category}_{i:02d}"
            
            instance_data = generate_instance(
                instance_id=instance_id,
                n_customers=n_customers,
                n_drones=params["k"],
                battery_km=params["battery"],
                payload_kg=params["payload"]
            )
            
            file_path = os.path.join(out_dir, f"{instance_id}.json")
            with open(file_path, 'w') as f:
                json.dump(instance_data, f, indent=2)
                
            print(f"Generated {file_path}")

if __name__ == "__main__":
    main()
