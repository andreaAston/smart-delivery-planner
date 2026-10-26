import numpy as np

from aco_routing import (
    aco_route,
    active_route_stops,
    build_distance_matrix,
    nearest_neighbor,
    tour_length,
)
from fuzzy_priority import add_priority, fuzzy_priority
from ga_loading import greedy_load, run_ga
from load_route import load_route, pick_routes


SEEDS = 10
MAIN_CAPACITY_FRACTION = 0.40


def add_priorities(packages):
    return add_priority(packages).reset_index(drop=True)


def build_plan(packages, stops, capacity_fraction=0.40, seed=0, rush=False):
    """Create one feasible load and an ACO tour for one route and seed."""
    route_packages = packages.copy()
    if rush:
        no_window = route_packages.index[~route_packages["has_tw"]][:5]
        if len(no_window) != 5:
            raise ValueError("Rush scenario requires at least five no-window packages.")
        route_packages.loc[no_window, "has_tw"] = True
        route_packages.loc[no_window, "hours_left"] = 1.0
        route_packages.loc[no_window, "window_h"] = 1.0
        route_packages.loc[no_window, "priority"] = [
            fuzzy_priority(1.0, 1.0)
        ] * len(no_window)

    priorities = route_packages["priority"].to_numpy(dtype=float)
    volumes = route_packages["volume_cm3"].to_numpy(dtype=float)
    total_volume = float(volumes.sum())
    capacity = total_volume * capacity_fraction
    chromosome = run_ga(
        priorities,
        volumes,
        capacity,
        seed=seed,
        pop_size=50,
        generations=60,
        repair=True,
        seed_with_greedy=True,
    )
    selected = route_packages.loc[chromosome == 1].copy()
    route_stops = active_route_stops(stops, selected)
    distances = build_distance_matrix(route_stops)
    tour, distance, _ = aco_route(
        distances, n_ants=10, iterations=20, seed=seed
    )
    nearest_tour = nearest_neighbor(distances)
    driver_tour = list(range(len(route_stops)))

    return {
        "packages": route_packages,
        "selected": selected,
        "capacity_cm3": capacity,
        "used_cm3": float(chromosome @ volumes),
        "priority": float(chromosome @ priorities),
        "greedy_priority": float(
            greedy_load(priorities, volumes, capacity) @ priorities
        ),
        "stops": route_stops,
        "distances": distances,
        "aco_tour": tour,
        "aco_km": distance,
        "nearest_km": tour_length(nearest_tour, distances),
        "driver_km": tour_length(driver_tour, distances),
    }


def summarize_runs(runs):
    columns = ("priority", "aco_km", "nearest_km", "driver_km")
    return {
        key: {
            "mean": float(np.mean([run[key] for run in runs])),
            "min": float(np.min([run[key] for run in runs])),
            "max": float(np.max([run[key] for run in runs])),
        }
        for key in columns
    }


def format_range(summary, key):
    digits = 1 if key == "priority" else 2
    return (
        f"{summary[key]['min']:.{digits}f}-"
        f"{summary[key]['max']:.{digits}f}"
    )


def select_routes():
    candidates = pick_routes(score="High", min_tw=15, n=200)
    selected = []
    stations = set()
    for candidate in candidates:
        route_id, station, _, _ = candidate
        if station not in stations:
            selected.append(candidate)
            stations.add(station)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise ValueError(
            "Could not find one main and two backup routes from distinct stations."
        )
    return selected


def print_results_table(rows, columns):
    print(" | ".join(columns))
    print("-" * (len(" | ".join(columns))))
    for row in rows:
        print(" | ".join(str(row[column]) for column in columns))


def run_step_four(packages, stops):
    print("\nSTEP 4: ACO parameter sweep on the main route")
    baseline = build_plan(packages, stops, MAIN_CAPACITY_FRACTION, seed=0)
    route_stops = baseline["stops"]
    distances = baseline["distances"]
    print(f"Same selected stops (including station): {len(route_stops)}.")
    rows = []
    beta_three_seed_zero = None
    for beta in (0, 1, 3, 5):
        lengths = [
            aco_route(
                distances,
                n_ants=10,
                iterations=30,
                beta=beta,
                seed=seed,
            )[1]
            for seed in range(SEEDS)
        ]
        if beta == 3:
            beta_three_seed_zero = lengths[0]
        rows.append(
            {
                "beta": beta,
                "mean_km": f"{np.mean(lengths):.2f}",
                "worst_km": f"{np.max(lengths):.2f}",
            }
        )
    print(
        f"Same-stop distances: driver {baseline['driver_km']:.2f} km; "
        f"nearest neighbour {baseline['nearest_km']:.2f} km; "
        f"ACO beta=3, seed 0 {beta_three_seed_zero:.2f} km."
    )
    print_results_table(rows, ["beta", "mean_km", "worst_km"])


def run_capacity_scenarios(packages, stops):
    print("\nSTEP 5.1: Replacement van capacity")
    rows = []
    for fraction in (0.30, 0.40, 0.60):
        runs = [
            build_plan(packages, stops, fraction, seed=seed)
            for seed in range(SEEDS)
        ]
        summary = summarize_runs(runs)
        rows.append(
            {
                "capacity": f"{fraction:.0%}",
                "mean_priority": f"{summary['priority']['mean']:.1f}",
                "priority_range": (
                    f"{summary['priority']['min']:.1f}-"
                    f"{summary['priority']['max']:.1f}"
                ),
                "mean_aco_km": f"{summary['aco_km']['mean']:.2f}",
                "aco_range_km": (
                    f"{summary['aco_km']['min']:.2f}-"
                    f"{summary['aco_km']['max']:.2f}"
                ),
            }
        )
    print_results_table(
        rows,
        [
            "capacity",
            "mean_priority",
            "priority_range",
            "mean_aco_km",
            "aco_range_km",
        ],
    )


def run_backup_scenarios(backups):
    print("\nSTEP 5.2: Backup routes from other stations")
    rows = []
    for route_id, station, _, _ in backups:
        packages, stops, info = load_route(route_id)
        packages = add_priorities(packages)
        runs = [
            build_plan(
                packages,
                stops,
                MAIN_CAPACITY_FRACTION,
                seed=seed,
            )
            for seed in range(SEEDS)
        ]
        summary = summarize_runs(runs)
        mean_greedy = float(np.mean([run["greedy_priority"] for run in runs]))
        rows.append(
            {
                "route": route_id[-8:],
                "station": station,
                "score": info["score"],
                "GA_mean_priority": f"{summary['priority']['mean']:.1f}",
                "GA_priority_range": format_range(summary, "priority"),
                "greedy_priority": f"{mean_greedy:.1f}",
                "ACO_mean_km": f"{summary['aco_km']['mean']:.2f}",
                "ACO_km_range": format_range(summary, "aco_km"),
                "NN_mean_km": f"{summary['nearest_km']['mean']:.2f}",
                "NN_km_range": format_range(summary, "nearest_km"),
            }
        )
    print_results_table(
        rows,
        [
            "route",
            "station",
            "score",
            "GA_mean_priority",
            "GA_priority_range",
            "greedy_priority",
            "ACO_mean_km",
            "ACO_km_range",
            "NN_mean_km",
            "NN_km_range",
        ],
    )


def run_rush_scenario(packages, stops):
    print("\nSTEP 5.3: Five no-window parcels receive a one-hour deadline")
    baseline_runs = [
        build_plan(packages, stops, MAIN_CAPACITY_FRACTION, seed=seed)
        for seed in range(SEEDS)
    ]
    rush_runs = [
        build_plan(
            packages,
            stops,
            MAIN_CAPACITY_FRACTION,
            seed=seed,
            rush=True,
        )
        for seed in range(SEEDS)
    ]
    bumped_counts = {}
    for baseline, rush in zip(baseline_runs, rush_runs):
        removed = set(baseline["selected"]["package"]) - set(
            rush["selected"]["package"]
        )
        for package_id in removed:
            bumped_counts[package_id] = bumped_counts.get(package_id, 0) + 1

    baseline_summary = summarize_runs(baseline_runs)
    rush_summary = summarize_runs(rush_runs)
    print(
        f"Baseline total priority: {baseline_summary['priority']['mean']:.1f} "
        f"(range {format_range(baseline_summary, 'priority')})"
    )
    print(
        f"Rush-order total priority: {rush_summary['priority']['mean']:.1f} "
        f"(range {format_range(rush_summary, 'priority')})"
    )
    print("Packages displaced from baseline load in at least one paired run:")
    if not bumped_counts:
        print("None.")
    else:
        for package_id, count in sorted(
            bumped_counts.items(), key=lambda item: (-item[1], item[0])
        )[:10]:
            print(f"  {package_id}: {count}/{SEEDS} runs")


def run_planner():
    selected_routes = select_routes()
    main_id, main_station, main_stops, main_tw = selected_routes[0]
    print("Selected High-score routes (at least 15 time-window packages):")
    print_results_table(
        [
            {
                "role": role,
                "route": candidate[0],
                "station": candidate[1],
                "stops": candidate[2],
                "time_windows": candidate[3],
            }
            for role, candidate in zip(
                ("main", "backup 1", "backup 2"), selected_routes
            )
        ],
        ["role", "route", "station", "stops", "time_windows"],
    )

    packages, stops, info = load_route(main_id)
    packages = add_priorities(packages)

    print("\nDISPATCH PLAN: main route, 40% replacement capacity, seed 0")
    plan = build_plan(packages, stops, MAIN_CAPACITY_FRACTION, seed=0)
    left_at_depot = packages.loc[
        ~packages["package"].isin(plan["selected"]["package"])
    ]
    stop_order = plan["stops"].iloc[plan["aco_tour"]]["stop"].tolist()
    print(f"Station: {main_station} ({info['station']})")
    print(
        f"Packages loaded: {len(plan['selected'])}; "
        f"left at depot: {len(left_at_depot)}"
    )
    print(
        f"Capacity used: {plan['used_cm3']:,.0f} / "
        f"{plan['capacity_cm3']:,.0f} cm3"
    )
    print(f"Selected delivery stop order (station first, return implied): {stop_order}")
    print(f"ACO distance: {plan['aco_km']:.2f} km")
    print(
        "Benchmark on the same selected stops: "
        f"driver {plan['driver_km']:.2f} km; "
        f"nearest neighbour {plan['nearest_km']:.2f} km."
    )
    print(f"Route checks: {main_stops} stops, {main_tw} time-window packages.")

    run_step_four(packages, stops)
    run_capacity_scenarios(packages, stops)
    run_backup_scenarios(selected_routes[1:])
    run_rush_scenario(packages, stops)


if __name__ == "__main__":
    run_planner()
