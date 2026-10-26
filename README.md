# Smart Delivery Planner

An assignment project that combines fuzzy urgency scoring, genetic-algorithm
package loading, and ant-colony route ordering for a smaller replacement van.

## Data

The loader expects these Amazon Last Mile Routing Research Challenge training
files under
`almrrc2021/almrrc2021-data-training/model_build_inputs/`:

- `route_data.json`
- `package_data.json`
- `actual_sequences.json`

Run `python download_data.py` to retrieve them if they are not already present.
The project does not download `travel_times.json`.

Dataset citation: Merchán, D. et al. (2022), *2021 Amazon Last Mile Routing
Research Challenge: Data Set*, Transportation Science,
doi:10.1287/trsc.2022.1173. Licence: CC BY-NC 4.0.

## Modules

- `load_route.py` loads one route, its stops and packages, computes package
  time-window features and volume, and provides Haversine distance and route
  selection helpers.
- `fuzzy_priority.py` provides the handout's Mamdani fuzzy rules (`v1`) and
  the revised `v2` rules used by default. It also includes the handout check
  values and `add_priority()` for package tables.
- `ga_loading.py` provides the priority-per-volume greedy baseline, a GA with
  the handout's penalty per 1,000 cm3, and optional greedy seeding and overload
  repair. Its command-line run reports 20-seed plain and improved comparisons.
- `aco_routing.py` provides the distance matrix, nearest-neighbour baseline
  and ACO. It routes through loaded-package stops, includes the station as
  node 0, compares methods over the same stops, and runs a 10-seed beta sweep.
- `planner.py` integrates the modules. It prints a dispatcher-ready plan and
  evaluates the 30%, 40% and 60% capacity cases, two backup stations, and five
  simulated rush orders using 10 seeds.
- `route_profile.py` plots a route and reports its full driver-loop benchmark.

## Run

From this folder:

```powershell
python planner.py
```

The components can also be run individually:

```powershell
python route_profile.py
python fuzzy_priority.py
python ga_loading.py
python aco_routing.py
```

## Interpretation

The supplied coordinates, package dimensions, time windows and driver stop
sequence are dataset observations. The fuzzy urgency rules, replacement-van
capacity fractions and rush-order scenario are assignment assumptions.
Reported route distances are straight-line Haversine distances, not road
distances or travel times. An ACO result is compared with the driver and
nearest-neighbour baselines only on the same stops selected for that load.
