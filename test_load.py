from load_route import pick_routes, load_route

# Find 5 candidate routes with a "High" score and at least 15 time-window packages
candidates = pick_routes(score="High", min_tw=15, n=5)
print("Found candidate routes (Route ID, Station, Stops, Time Windows):")
for c in candidates:
    print(c)

# Pick the first one as our main route ID
if candidates:
    main_route_id = candidates[0][0]
    pkgs, stops, info = load_route(main_route_id)
    print(f"\nSuccessfully loaded main route: {main_route_id}")
    print(f"Station: {info['station']}")
    print(f"Total Stops: {len(stops)}")
    print(f"Total Packages: {len(pkgs)}")
    print(f"Van Capacity: {info['capacity_cm3']:,} cm3")

candidates = pick_routes(score="High", min_tw=15, n=1)
if candidates:
    route_id = candidates[0][0]
    pkgs, stops, info = load_route(route_id)

    # 1. Packages with time windows
    tw_count = pkgs["has_tw"].sum()  # True counts as 1, False as 0

    # 2. Total volume and replacement van capacity (40%)[cite: 14]
    total_vol = pkgs["volume_cm3"].sum()
    replacement_cap = total_vol * 0.40

    print(f"Route ID: {route_id}")
    print(f"Packages with a Time Window: {tw_count}")
    print(f"Total Package Volume: {total_vol:,.2f} cm³")
    print(f"Replacement Van Capacity (40%): {replacement_cap:,.2f} cm³")
