import math
import numpy as np
import pygad
import matplotlib.pyplot as plt

# Coordinates (first = depot)
coords = np.array([
    [30, 30],  # depot
    [37, 52],  # port 1
    [49, 49],  # port 2
    [52, 64],  # port 3
    [31, 62],  # port 4
    [52, 33],  # port 5
    [42, 41],  # port 6
    [52, 41],  # port 7
    [57, 58]   # port 8
], dtype=float)

# Ship fuel & emission parameters
fuel_rate_per_km = 0.25  # liters/km
emission_factor = 3.206  # kg CO₂ per liter

customer_ids = np.arange(1, len(coords))

# Distance matrix
def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

N = len(coords)
dist = np.zeros((N, N), dtype=float)
for i in range(N):
    for j in range(N):
        dist[i, j] = euclid(coords[i], coords[j])

# Decode GA solution into route order
def decode_solution(keys):
    order = customer_ids[np.argsort(keys)]
    return order.tolist()

# Route metrics
def route_distance(route):
    if not route:
        return 0.0
    total = dist[0, route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i], route[i+1]]
    total += dist[route[-1], 0]
    return total

def route_cost(route):
    dist_km = route_distance(route)
    fuel = dist_km * fuel_rate_per_km
    co2 = fuel * emission_factor
    return dist_km, fuel, co2

# Fitness function (minimize fuel)
def fitness_func(ga_instance, solution, solution_idx):
    route = decode_solution(solution)
    dist_km, fuel, co2 = route_cost(route)
    return -fuel

# GA config
num_genes = len(customer_ids)
ga = pygad.GA(
    num_generations=200,
    sol_per_pop=40,
    num_parents_mating=20,
    fitness_func=fitness_func,
    num_genes=num_genes,
    gene_type=float,
    gene_space={'low': 0.0, 'high': 1.0},
    parent_selection_type="tournament",
    K_tournament=3,
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=20,
    random_mutation_min_val=0.0,
    random_mutation_max_val=1.0,
    keep_parents=2,
)

# Tracking
best_distances = []
best_fuels = []
best_co2s = []

def on_generation(ga_instance):
    sol, fit, _ = ga_instance.best_solution()
    route = decode_solution(sol)
    dist_km, fuel, co2 = route_cost(route)

    best_distances.append(dist_km)
    best_fuels.append(fuel)
    best_co2s.append(co2)

    if (ga_instance.generations_completed % 20) == 0 or ga_instance.generations_completed == 1:
        print(f"Gen {ga_instance.generations_completed:3d} | Dist={dist_km:.2f} km | Fuel={fuel:.2f} L | CO₂={co2:.2f} kg")

ga.on_generation = on_generation

# Run GA
ga.run()

# Best result
best_keys, best_fitness, _ = ga.best_solution()
best_route = decode_solution(best_keys)
dist_km, fuel, co2 = route_cost(best_route)

print("\nBest TSP route (customer IDs):", best_route)
print(f"Distance: {dist_km:.2f} km")
print(f"Fuel: {fuel:.2f} liters")
print(f"CO₂: {co2:.2f} kg")

# Plot fitness and metrics
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

axs[0].plot(best_distances, label="Distance (km)")
axs[0].set_ylabel("Distance (km)")
axs[0].legend()

axs[1].plot(best_fuels, color="orange", label="Fuel (liters)")
axs[1].set_ylabel("Fuel (liters)")
axs[1].legend()

axs[2].plot(best_co2s, color="green", label="CO₂ (kg)")
axs[2].set_ylabel("CO₂ (kg)")
axs[2].set_xlabel("Generation")
axs[2].legend()

plt.tight_layout()
plt.show()
