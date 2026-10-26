import matplotlib.pyplot as plt
from load_route import load_route, km, pick_routes

# 1. Pick and load your High-score route with time windows
candidates = pick_routes(score="High", min_tw=15, n=1)
if not candidates:
    raise ValueError(
        "No matching routes found! Try adjusting your pick_routes filters."
    )

route_id = candidates[0][0]
pkgs, stops, info = load_route(route_id)

# 2. Sort stops in the exact order the human driver visited them
sorted_stops = stops.sort_values("driver_order").reset_index(drop=True)

# 3. Calculate the Human Driver's Benchmark Distance (closing the loop back to start)
latitudes = sorted_stops["lat"].values
longitudes = sorted_stops["lng"].values
n_stops = len(sorted_stops)

driver_distance = 0.0
for i in range(n_stops):
    lat1, lng1 = latitudes[i], longitudes[i]
    lat2, lng2 = latitudes[(i + 1) % n_stops], longitudes[(i + 1) % n_stops]
    driver_distance += km(lat1, lng1, lat2, lng2)

print(f"--- Route Profile for {route_id} ---")
print(f"Station Code: {info['station']}")
print(f"Total Stops: {n_stops}")
print(f"Human Driver Benchmark Distance: {driver_distance:.2f} km\n")

# 4. Identify which stops have a time-window package (fixed redundant comparison)
stops_with_tw = set(pkgs[pkgs["has_tw"]]["stop"])
sorted_stops["has_window"] = sorted_stops["stop"].isin(stops_with_tw)

# 5. Plot the Map (Longitude on X, Latitude on Y)
plt.figure(figsize=(10, 8))

tw_stops = sorted_stops[sorted_stops["has_window"]]
regular_stops = sorted_stops[~sorted_stops["has_window"]]

# Scatter plots for stops
plt.scatter(
    regular_stops["lng"],
    regular_stops["lat"],
    color="cornflowerblue",
    label="Regular Stops",
    alpha=0.7,
    s=40,
)
plt.scatter(
    tw_stops["lng"],
    tw_stops["lat"],
    color="crimson",
    label="Time-Window Stops",
    s=70,
    zorder=5,
)

# Draw lines showing the human driver's path sequence
plt.plot(
    sorted_stops["lng"],
    sorted_stops["lat"],
    color="gray",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="Human Driver Path",
)

plt.title(
    f"Amazon Route Profile: {route_id} ({info['station']})\nBenchmark Distance: {driver_distance:.2f} km",
    fontsize=12,
)
plt.xlabel("Longitude (X)")
plt.ylabel("Latitude (Y)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()
