import json
import math
from datetime import datetime
import pandas as pd

DATA = "almrrc2021/almrrc2021-data-training/model_build_inputs/"


def load_route(route_id, data_dir=DATA):
    """Return (packages DataFrame, stops DataFrame, route info) for one route."""
    routes = json.load(open(data_dir + "route_data.json"))
    pkgs = json.load(open(data_dir + "package_data.json"))
    seqs = json.load(open(data_dir + "actual_sequences.json"))

    r = routes[route_id]
    depart = datetime.fromisoformat(f"{r['date_YYYY_MM_DD']} {r['departure_time_utc']}")

    stops = pd.DataFrame([{"stop": s, **v} for s, v in r["stops"].items()])
    stops["driver_order"] = stops["stop"].map(seqs[route_id]["actual"])

    rows = []
    for stop, packages in pkgs[route_id].items():
        for pid, p in packages.items():
            tw, dim = p["time_window"], p["dimensions"]
            has_tw = isinstance(tw["end_time_utc"], str)
            if has_tw:
                end = datetime.fromisoformat(tw["end_time_utc"])
                start = datetime.fromisoformat(tw["start_time_utc"])
                hours_left = max((end - depart).total_seconds() / 3600, 0)
                width = (end - start).total_seconds() / 3600
            else:
                # no window: deliver any time today
                hours_left, width = 24.0, 24.0
            rows.append(
                {
                    "package": pid,
                    "stop": stop,
                    "has_tw": has_tw,
                    "hours_left": round(hours_left, 2),
                    "window_h": round(width, 2),
                    "volume_cm3": dim["depth_cm"] * dim["height_cm"] * dim["width_cm"],
                }
            )

    packages = pd.DataFrame(rows).merge(stops[["stop", "lat", "lng"]], on="stop")
    info = {
        "station": r["station_code"],
        "capacity_cm3": r["executor_capacity_cm3"],
        "score": r["route_score"],
        "depart": depart,
    }
    return packages, stops, info


def km(lat1, lng1, lat2, lng2):
    """Great-circle (haversine) distance in km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lng2 - lng1)
    dp = math.radians(lat2 - lat1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def pick_routes(data_dir=DATA, min_tw=15, score="High", n=5):
    """List routes with a good driver score and at least min_tw time-window packages."""
    routes = json.load(open(data_dir + "route_data.json"))
    pkgs = json.load(open(data_dir + "package_data.json"))
    found = []
    for rid, r in routes.items():
        if r["route_score"] != score:
            continue
        n_tw = sum(
            isinstance(p["time_window"]["end_time_utc"], str)
            for stop in pkgs[rid].values()
            for p in stop.values()
        )
        if n_tw > min_tw:
            found.append((rid, r["station_code"], len(r["stops"]), n_tw))
            if len(found) == n:
                break
    return found
