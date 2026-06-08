# Complexity Analysis
The Drone Delivery Optimization problem is a variant of the Capacitated Vehicle Routing Problem (CVRP).
We prove its NP-hardness as follows:

1. The Traveling Salesman Problem (TSP) is NP-hard (well established).
2. CVRP is a generalization of TSP: set K=1 vehicle/drone with unlimited battery and payload. Then finding the optimal drone route equals finding the optimal TSP tour.
3. Since TSP reduces polynomially to Drone Delivery (by setting K=1, B=infinity, Q=infinity), and TSP is NP-hard, Drone Delivery is also NP-hard.
4. The decision version ("is there a feasible set of routes with total cost <= C?") is in NP because a solution can be verified in polynomial time.
Therefore the Drone Delivery Optimization problem is NP-complete in its decision form and NP-hard in its optimization form.
