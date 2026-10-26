import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ga_loading import priority_table, run_ga
from load_route import km, load_route, pick_routes


def build_distance_matrix(stops):
    coords = stops[["lat", "lng"]].to_numpy(dtype=float)
    n_stops = len(coords)
    distances = np.zeros((n_stops, n_stops), dtype=float)
    for i in range(n_stops):
        for j in range(i + 1, n_stops):
            distance = km(
                coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]
            )
            distances[i, j] = distance
            distances[j, i] = distance
    return distances


def tour_length(tour, distances):
    if len(tour) < 2:
        return 0.0
    return float(
        sum(
            distances[tour[i], tour[(i + 1) % len(tour)]]
            for i in range(len(tour))
        )
    )


def nearest_neighbor(distances, start=0):
    n_stops = len(distances)
    if n_stops == 0:
        raise ValueError("At least one stop is required.")
    if not 0 <= start < n_stops:
        raise ValueError("Start index is outside the distance matrix.")

    tour = [start]
    unvisited = set(range(n_stops))
    unvisited.remove(start)
    while unvisited:
        current = tour[-1]
        next_stop = min(
            unvisited,
            key=lambda node: (distances[current, node], node),
        )
        tour.append(next_stop)
        unvisited.remove(next_stop)
    return tour


def aco_route(
    distances,
    n_ants=20,
    iterations=50,
    alpha=1.0,
    beta=3.0,
    evaporation=0.5,
    deposit_scale=1.0,
    seed=0,
):
    """Find a round-trip tour starting and ending at matrix index zero."""
    n_stops = len(distances)
    if n_stops == 0:
        raise ValueError("At least one stop is required.")
    if n_ants < 1 or iterations < 1:
        raise ValueError("ACO ant and iteration counts must be positive.")
    if not 0 <= evaporation < 1:
        raise ValueError("Evaporation must be in the interval [0, 1).")
    if n_stops == 1:
        return [0], 0.0, [0.0] * iterations

    rng = np.random.default_rng(seed)
    visibility = np.zeros_like(distances, dtype=float)
    np.divide(1.0, distances, out=visibility, where=distances > 0)
    heuristic = visibility**beta
    pheromones = np.ones((n_stops, n_stops), dtype=float)
    best_tour = None
    best_distance = float("inf")
    history = []

    for _ in range(iterations):
        tours = []
        for _ in range(n_ants):
            tour = [0]
            visited = np.zeros(n_stops, dtype=bool)
            visited[0] = True
            for _ in range(n_stops - 1):
                candidates = np.flatnonzero(~visited)
                current = tour[-1]
                weights = (
                    pheromones[current, candidates] ** alpha
                    * heuristic[current, candidates]
                )
                if not np.isfinite(weights).all() or weights.sum() <= 0:
                    next_stop = int(rng.choice(candidates))
                else:
                    probabilities = weights / weights.sum()
                    next_stop = int(rng.choice(candidates, p=probabilities))
                tour.append(next_stop)
                visited[next_stop] = True

            distance = tour_length(tour, distances)
            tours.append((tour, distance))
            if distance < best_distance:
                best_tour = tour
                best_distance = distance

        pheromones *= 1.0 - evaporation
        for tour, distance in tours:
            if distance <= 0:
                continue
            deposit = deposit_scale / distance
            for index in range(n_stops):
                a = tour[index]
                b = tour[(index + 1) % n_stops]
                pheromones[a, b] += deposit
                pheromones[b, a] += deposit
        history.append(best_distance)

    return best_tour, best_distance, history


def active_route_stops(stops, selected_packages):
    active_ids = set(selected_packages["stop"].unique())
    station_rows = stops[stops["type"] == "Station"]
    if len(station_rows) != 1:
        raise ValueError(
            f"Expected exactly one station stop, found {len(station_rows)}."
        )
    station = station_rows.iloc[0]
    deliveries = stops[
        stops["stop"].isin(active_ids) & (stops["type"] != "Station")
    ]
    deliveries = deliveries.sort_values("driver_order")
    return pd.concat(
        [station.to_frame().T, deliveries], ignore_index=True
    ).reset_index(drop=True)


def run_aco_routing():
    candidates = pick_routes(score="High", min_tw=15, n=1)
    if not candidates:
        raise ValueError("No matching routes found.")

    route_id = candidates[0][0]
    packages, stops, _ = load_route(route_id)
    packages = priority_table(packages).reset_index(drop=True)
    priorities = packages["priority"].to_numpy(dtype=float)
    volumes = packages["volume_cm3"].to_numpy(dtype=float)
    capacity = float(volumes.sum() * 0.40)
    chromosome = run_ga(
        priorities,
        volumes,
        capacity,
        seed=42,
        repair=True,
        seed_with_greedy=True,
    )

    selected_packages = packages.loc[chromosome == 1]
    route_stops = active_route_stops(stops, selected_packages)
    distances = build_distance_matrix(route_stops)
    human_tour = list(range(len(route_stops)))
    nn_tour = nearest_neighbor(distances)
    beta_values = (0, 1, 3, 5)
    seeds = range(10)

    print(f"--- ACO Routing for Route {route_id} ---")
    print(f"Selected packages: {len(selected_packages)}")
    print(f"Active delivery stops plus station: {len(route_stops)}")
    print("All comparisons use the same selected stops and station start.")
    print(f"Filtered driver distance: {tour_length(human_tour, distances):.2f} km")
    print(f"Nearest-neighbour distance: {tour_length(nn_tour, distances):.2f} km")
    print("\nACO beta sweep (10 seeds each; mean and worst distance):")

    beta_results = {}
    for beta in beta_values:
        runs = [
            aco_route(
                distances,
                n_ants=10,
                iterations=30,
                beta=beta,
                seed=seed,
            )
            for seed in seeds
        ]
        lengths = np.array([result[1] for result in runs])
        beta_results[beta] = runs
        print(
            f"beta={beta}: mean {lengths.mean():.2f} km, "
            f"worst {lengths.max():.2f} km"
        )

    aco_tour, aco_distance, _ = beta_results[3][0]
    print("\nSame-stop route comparison:")
    print(f"Driver:           {tour_length(human_tour, distances):.2f} km")
    print(f"Nearest neighbour: {tour_length(nn_tour, distances):.2f} km")
    print(f"ACO (beta=3):     {aco_distance:.2f} km")

    coords = route_stops[["lat", "lng"]].to_numpy(dtype=float)
    human_closed = coords[human_tour + [human_tour[0]]]
    aco_closed = coords[aco_tour + [aco_tour[0]]]
    plt.figure(figsize=(10, 8))
    plt.plot(
        human_closed[:, 1],
        human_closed[:, 0],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        marker=".",
        label=f"Driver, selected stops ({tour_length(human_tour, distances):.2f} km)",
    )
    plt.plot(
        aco_closed[:, 1],
        aco_closed[:, 0],
        color="forestgreen",
        linewidth=2,
        marker="o",
        label=f"ACO beta=3 ({aco_distance:.2f} km)",
    )
    plt.scatter(coords[0, 1], coords[0, 0], color="black", marker="s", s=75, label="Station")
    plt.title(f"Smart Delivery Planner: ACO vs Driver ({route_id})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_aco_routing()
