"""Mamdani fuzzy priority scores for delivery-package time windows."""

import numpy as np
import matplotlib.pyplot as plt

from load_route import load_route, pick_routes


def tri(x, a, b, c):
    """Triangle membership function."""
    return max(
        min(
            (x - a) / (b - a) if b != a else 0,
            (c - x) / (c - b) if c != b else 0,
        ),
        0.0,
    )


def trap(x, a, b, c, d):
    """Trapezoid membership function with shoulder support."""
    up = 1.0 if b == a else (x - a) / (b - a)
    down = 1.0 if d == c else (d - x) / (d - c)
    return max(min(up, 1.0, down), 0.0)


HOURS = {
    "Soon": lambda h: trap(h, 0, 0, 1.5, 3),
    "Midday": lambda h: tri(h, 2, 4.5, 7),
    "Later": lambda h: trap(h, 6, 9, 24, 24),
}
WIDTH = {
    "Narrow": lambda w: trap(w, 0, 0, 2, 4),
    "Normal": lambda w: tri(w, 3, 6, 9),
    "Wide": lambda w: trap(w, 8, 12, 24, 24),
}

P = np.linspace(0, 100, 1001)
PRIO = {
    "Low": np.array([tri(p, 0, 20, 40) for p in P]),
    "Medium": np.array([tri(p, 30, 50, 70) for p in P]),
    "High": np.array([tri(p, 60, 80, 100) for p in P]),
}


def priority(hours_left, window_h, rules="v2", verbose=False):
    """Return a 0-100 urgency score using the selected Mamdani rule base."""
    hours_membership = {key: term(hours_left) for key, term in HOURS.items()}
    width_membership = {key: term(window_h) for key, term in WIDTH.items()}

    fired = [
        (
            min(
                hours_membership["Soon"],
                max(width_membership["Narrow"], width_membership["Normal"]),
            ),
            "High",
        ),
        (min(hours_membership["Soon"], width_membership["Wide"]), "Medium"),
        (min(hours_membership["Midday"], width_membership["Narrow"]), "High"),
        (min(hours_membership["Midday"], width_membership["Normal"]), "Medium"),
        (min(hours_membership["Midday"], width_membership["Wide"]), "Low"),
    ]
    if rules == "v1":
        fired.append((hours_membership["Later"], "Low"))
    elif rules == "v2":
        fired.extend(
            [
                (
                    min(hours_membership["Later"], width_membership["Wide"]),
                    "Low",
                ),
                (
                    min(
                        hours_membership["Later"],
                        max(
                            width_membership["Narrow"],
                            width_membership["Normal"],
                        ),
                    ),
                    "Medium",
                ),
            ]
        )
    else:
        raise ValueError("rules must be either 'v1' or 'v2'.")

    aggregate = np.zeros_like(P)
    for strength, output_term in fired:
        aggregate = np.maximum(
            aggregate, np.minimum(strength, PRIO[output_term])
        )
    if verbose:
        print("Rule strengths:", [round(strength, 3) for strength, _ in fired])
    return (
        float((aggregate * P).sum() / aggregate.sum())
        if aggregate.sum() > 0
        else 0.0
    )


def fuzzy_priority(hours_left, window_h):
    """Backward-compatible v2 priority function used by the solution scripts."""
    return round(priority(hours_left, window_h, rules="v2"), 1)


def add_priority(packages, rules="v2"):
    """Return the package table with a priority column, scoring each unique pair once."""
    output = packages.drop(columns=["priority"], errors="ignore").copy()
    pairs = output[["hours_left", "window_h"]].drop_duplicates().copy()
    pairs["priority"] = [
        priority(hours, width, rules=rules)
        for hours, width in zip(pairs["hours_left"], pairs["window_h"])
    ]
    return output.merge(pairs, on=["hours_left", "window_h"], how="left")


if __name__ == "__main__":
    checks = [
        ((5.4, 7), 50.0),
        ((2.5, 5), 68.4),
        ((1, 1), 80.0),
        ((24, 24), 20.0),
    ]
    for rule_set in ("v1", "v2"):
        for (hours, width), expected in checks:
            actual = priority(hours, width, rules=rule_set)
            assert abs(actual - expected) < 0.05, (
                rule_set,
                hours,
                width,
                actual,
                expected,
            )
        print(f"rules={rule_set}: all four handout check values match")

    for hours, width in ((6.96, 7), (7.96, 7), (7.96, 8), (9.96, 11)):
        print(
            f"({hours}, {width}): v1={priority(hours, width, 'v1'):.1f}, "
            f"v2={priority(hours, width, 'v2'):.1f}"
        )

    candidates = pick_routes(score="High", min_tw=15, n=1)
    if candidates:
        route_id = candidates[0][0]
        packages, _, _ = load_route(route_id)
        packages = add_priority(packages)
        print(f"\nProcessed {len(packages)} packages for route {route_id}.")
        print(
            packages[
                ["package", "hours_left", "window_h", "priority"]
            ].head(10)
        )
        plt.figure(figsize=(8, 5))
        plt.hist(
            packages["priority"],
            bins=15,
            color="teal",
            edgecolor="black",
            alpha=0.8,
        )
        plt.title(f"Package Priority Distribution for Route {route_id}")
        plt.xlabel("Fuzzy Priority Score")
        plt.ylabel("Number of Packages")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.show()
