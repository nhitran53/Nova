"""
NOVA shared simulation rules.

The one place that defines how fast oxygen is used. The simulator (tank
level) and the API (route check) both import from here, so the formula
can't drift apart between two copies. Predictions can still differ from
what actually happens: they depend on the activity, distance and
conditions assumed at the time of the check.

These are demo assumptions, not physiological models.
"""

import math

# Multipliers on missions.o2_burn_pct_per_min
ACTIVITY_O2 = {"resting": 0.6, "walking": 1.0, "working": 1.3, "running": 1.8}
SCENARIO_O2 = {"normal": 1.0, "o2_leak": 6.0, "dust_storm": 1.1, "radiation_event": 1.0}


def _finite(*values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def o2_burn_pct_per_min(mission, activity=None, scenario=None) -> float:
    """Percentage POINTS of tank used per simulated minute.

    mission  : dict-like row with o2_burn_pct_per_min, activity, scenario
    activity : override (e.g. 'walking' for travel, 'working' at a site)
    scenario : override
    """
    a = activity or mission["activity"]
    s = scenario or mission["scenario"]
    if a not in ACTIVITY_O2:
        raise ValueError(f"Unknown activity: {a!r}")
    if s not in SCENARIO_O2:
        raise ValueError(f"Unknown scenario: {s!r}")
    base = float(mission["o2_burn_pct_per_min"])
    if not _finite(base) or base <= 0:
        raise ValueError("o2_burn_pct_per_min must be a finite number greater than zero.")
    return base * ACTIVITY_O2[a] * SCENARIO_O2[s]


def o2_used_pct(mission, sim_seconds: float, activity=None, scenario=None) -> float:
    """Tank percentage points used over a span of simulated time."""
    if not _finite(sim_seconds) or sim_seconds < 0:
        raise ValueError("sim_seconds must be a finite number, zero or greater.")
    return o2_burn_pct_per_min(mission, activity, scenario) * sim_seconds / 60.0


def estimate_route_o2(mission, outbound_m: float, return_m: float,
                      minutes_at_destination: float = 0.0,
                      destination_activity: str = "working") -> dict:
    """Oxygen budget for: walk there, spend time there, walk back.

    travel_minutes = (outbound + return) / walk_speed / 60
    required_pct   = travel_minutes      x burn(walking)
                   + destination_minutes x burn(destination_activity)
                   + o2_reserve_pct        # reserve = percentage POINTS kept in the tank

    Uses the mission's current scenario (a leak makes every minute cost more).
    For navigation_requests.assessed_o2_burn_pct_per_min, store travel_burn_pct_per_min.
    """
    walk_speed = float(mission["walk_speed_mps"])
    reserve = float(mission["o2_reserve_pct"])

    if not _finite(outbound_m, return_m, minutes_at_destination, walk_speed, reserve):
        raise ValueError("Route inputs must be finite numbers.")
    if outbound_m < 0 or return_m < 0 or minutes_at_destination < 0:
        raise ValueError("Distances and destination time cannot be negative.")
    if walk_speed <= 0:
        raise ValueError("Walking speed must be greater than zero.")
    if not 0 <= reserve <= 100:
        raise ValueError("Oxygen reserve must be between 0 and 100.")

    travel_burn = o2_burn_pct_per_min(mission, activity="walking")
    dest_burn = o2_burn_pct_per_min(mission, activity=destination_activity)
    travel_minutes = (outbound_m + return_m) / walk_speed / 60.0
    travel_pct = travel_minutes * travel_burn
    dest_pct = minutes_at_destination * dest_burn

    return {
        "travel_minutes": travel_minutes,
        "minutes_at_destination": minutes_at_destination,
        "trip_minutes": travel_minutes + minutes_at_destination,
        "travel_burn_pct_per_min": travel_burn,
        "destination_burn_pct_per_min": dest_burn,
        "destination_activity": destination_activity,
        "travel_o2_pct": travel_pct,
        "destination_o2_pct": dest_pct,
        "reserve_pct": reserve,
        "walk_speed_mps": walk_speed,
        "required_pct": travel_pct + dest_pct + reserve,  # INCLUDES reserve
    }