import numpy as np
import pandas as pd

from fuzzy_priority import add_priority
from load_route import load_route, pick_routes


def greedy_load(priorities, volumes, capacity):
    """Pack by priority per cm3, keeping every selected load within capacity."""
    ratios = priorities / np.maximum(volumes, 1.0)
    order = np.argsort(-ratios, kind="stable")
    chromosome = np.zeros(len(priorities), dtype=np.int8)
    used = 0.0
    for index in order:
        if used + volumes[index] <= capacity:
            chromosome[index] = 1
            used += volumes[index]
    return chromosome


def repair_chromosome(chromosome, priorities, volumes, capacity):
    """Remove the lowest priority-per-volume packages until the load fits."""
    repaired = chromosome.copy()
    loaded = np.flatnonzero(repaired)
    excess = float(volumes[loaded].sum() - capacity)
    if excess <= 0:
        return repaired

    ratios = priorities / np.maximum(volumes, 1.0)
    order = loaded[np.argsort(ratios[loaded], kind="stable")]
    for index in order:
        repaired[index] = 0
        excess -= volumes[index]
        if excess <= 0:
            break
    return repaired


def fitness(chromosome, priorities, volumes, capacity, penalty_per_1000=10.0):
    total_volume = float(chromosome @ volumes)
    total_priority = float(chromosome @ priorities)
    excess = max(0.0, total_volume - capacity)
    return total_priority - penalty_per_1000 * excess / 1000.0


def run_ga(
    priorities,
    volumes,
    capacity,
    seed=0,
    pop_size=100,
    generations=200,
    crossover_rate=0.8,
    mutation_rate=0.01,
    penalty_per_1000=10.0,
    repair=False,
    seed_with_greedy=False,
):
    """Run one loading GA and return its best chromosome."""
    if len(priorities) != len(volumes):
        raise ValueError("Priorities and volumes must have the same length.")
    if len(priorities) < 2:
        raise ValueError("The GA requires at least two packages.")
    if capacity < 0 or pop_size < 2 or generations < 1:
        raise ValueError("Capacity must be non-negative; GA sizes must be positive.")

    rng = np.random.default_rng(seed)
    n_items = len(priorities)
    population = (rng.random((pop_size, n_items)) < 0.30).astype(np.int8)
    greedy = greedy_load(priorities, volumes, capacity)
    if seed_with_greedy:
        population[0] = greedy
    if repair:
        population = np.asarray(
            [
                repair_chromosome(c, priorities, volumes, capacity)
                for c in population
            ],
            dtype=np.int8,
        )

    for _ in range(generations):
        scores = np.array(
            [
                fitness(c, priorities, volumes, capacity, penalty_per_1000)
                for c in population
            ]
        )
        elite = population[int(scores.argmax())].copy()
        contestant_indices = rng.integers(0, pop_size, (2 * pop_size, 3))
        winners = contestant_indices[
            np.arange(2 * pop_size),
            scores[contestant_indices].argmax(axis=1),
        ]
        parents = population[winners]
        parent_a, parent_b = parents[:pop_size], parents[pop_size:]

        cuts = rng.integers(1, n_items, pop_size)[:, None]
        crossover_mask = (
            (rng.random(pop_size) < crossover_rate)[:, None]
            & (np.arange(n_items)[None, :] >= cuts)
        )
        children = np.where(crossover_mask, parent_b, parent_a)
        mutations = rng.random((pop_size, n_items)) < mutation_rate
        children = np.where(mutations, 1 - children, children).astype(np.int8)

        if repair:
            children = np.asarray(
                [
                    repair_chromosome(c, priorities, volumes, capacity)
                    for c in children
                ],
                dtype=np.int8,
            )
        children[0] = elite
        population = children

    scores = np.array(
        [
            fitness(c, priorities, volumes, capacity, penalty_per_1000)
            for c in population
        ]
    )
    return population[int(scores.argmax())].copy()


def evaluate_runs(
    priorities,
    volumes,
    capacity,
    seeds=20,
    repair=False,
    seed_with_greedy=False,
    pop_size=100,
    generations=200,
):
    rows = []
    for seed in range(seeds):
        chromosome = run_ga(
            priorities,
            volumes,
            capacity,
            seed=seed,
            repair=repair,
            seed_with_greedy=seed_with_greedy,
            pop_size=pop_size,
            generations=generations,
        )
        total_volume = float(chromosome @ volumes)
        rows.append(
            {
                "seed": seed,
                "priority": float(chromosome @ priorities),
                "volume_cm3": total_volume,
                "overweight": total_volume > capacity + 1e-6,
                "packages": int(chromosome.sum()),
            }
        )
    return pd.DataFrame(rows)


def priority_table(packages):
    return add_priority(packages)


def run_ga_loading():
    candidates = pick_routes(score="High", min_tw=15, n=1)
    if not candidates:
        raise ValueError("No matching routes found.")

    route_id = candidates[0][0]
    packages, _, _ = load_route(route_id)
    packages = priority_table(packages)
    priorities = packages["priority"].to_numpy(dtype=float)
    volumes = packages["volume_cm3"].to_numpy(dtype=float)
    capacity = float(volumes.sum() * 0.40)
    greedy = greedy_load(priorities, volumes, capacity)
    greedy_priority = float(greedy @ priorities)

    plain = evaluate_runs(priorities, volumes, capacity)
    improved = evaluate_runs(
        priorities,
        volumes,
        capacity,
        repair=True,
        seed_with_greedy=True,
    )

    print(f"--- GA Loading for Route {route_id} ---")
    print(f"Packages: {len(packages)}")
    print(f"Total volume: {volumes.sum():,.2f} cm3")
    print(f"Replacement van capacity (40%): {capacity:,.2f} cm3")
    print(f"Greedy baseline priority: {greedy_priority:.1f}")
    print("GA results across 20 seeds (total priority):")
    for name, runs in (("Plain GA", plain), ("Seed + repair", improved)):
        print(
            f"{name}: best {runs['priority'].max():.1f}, "
            f"mean {runs['priority'].mean():.1f}, "
            f"worst {runs['priority'].min():.1f}, "
            f"overweight {int(runs['overweight'].sum())}/20"
        )
    print(
        "Packages with fewer than 3 hours left: "
        f"{int((packages['hours_left'] < 3).sum())}"
    )


if __name__ == "__main__":
    run_ga_loading()
