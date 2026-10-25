"""In-memory demo meteor state; all geometry is on the simulator x/z plane."""
import json
import math
import threading
import time
from pathlib import Path


def load_hazard_policy():
    cfg = json.loads(Path(__file__).with_name("turn_policy.json").read_text())["hazard_zone"]
    if not cfg["status"].startswith("PLACEHOLDER"):
        raise ValueError("hazard_zone must be marked PLACEHOLDER")
    return cfg


def inactive_hazard():
    return dict(active=False, kind="meteor", center_x_m=None, center_z_m=None,
                radius_m=None, eta_s=None, inside=False, edge_distance_m=None)


def nearest_safe_point(position, cfg):
    """Radially nearest point beyond the margin; at the center choose +x deterministically."""
    dx, dz = position["x_m"] - cfg["center_x_m"], position["z_m"] - cfg["center_z_m"]
    distance = math.hypot(dx, dz)
    ux, uz = (dx / distance, dz / distance) if distance else (1.0, 0.0)
    travel = max(0.0, cfg["radius_m"] + cfg["safe_margin_m"] - distance)
    return {"safe_place": {"x_m": position["x_m"] + ux * travel, "z_m": position["z_m"] + uz * travel},
            "safe_distance_m": travel, "safe_direction": {"x": ux, "z": uz}}


class HazardDatasource:
    """Datasource hook: subclasses provide hazard_position(). No database writes."""

    def _init_hazard(self, clock=time.monotonic):
        self._hazard_lock = threading.RLock()
        self._hazard_clock = clock
        self._hazard_started = None
        self.hazard_policy = load_hazard_policy()

    def set_demo_meteor(self, active):
        """Return True only on an inactive-to-active transition; repeated triggers do not reset ETA."""
        if not isinstance(active, bool):
            raise ValueError("active must be boolean")
        with self._hazard_lock:
            triggered = active and self._hazard_started is None
            if triggered:
                self._hazard_started = self._hazard_clock()
            elif not active:
                self._hazard_started = None
            return triggered

    def _hazard_status(self, position):
        """Build active status from one position while the hazard lock is held."""
        cfg = self.hazard_policy
        distance = None if position is None else math.hypot(
            position["x_m"] - cfg["center_x_m"], position["z_m"] - cfg["center_z_m"])
        return dict(active=True, kind="meteor", center_x_m=cfg["center_x_m"], center_z_m=cfg["center_z_m"],
                    radius_m=cfg["radius_m"], eta_s=max(0, math.ceil(cfg["eta_hours"] * 3600 -
                    (self._hazard_clock() - self._hazard_started))),
                    inside=None if distance is None else distance <= cfg["radius_m"],
                    edge_distance_m=None if distance is None else round(abs(distance - cfg["radius_m"])))

    def hazard_status(self):
        from .datasource import DataUnavailableError
        with self._hazard_lock:
            if self._hazard_started is None:
                return inactive_hazard()
            try:
                position = self.hazard_position()
            except DataUnavailableError:
                position = None
            return self._hazard_status(position)

    def hazard_snapshot(self):
        from .datasource import DataUnavailableError
        hazard = self.hazard_status()
        if hazard["active"] and hazard["inside"] is None:
            raise DataUnavailableError("meteor geometry requires an astronaut position")
        return {"meteor_inside": hazard["inside"]} if hazard["active"] else {}

    def hazard_tool_data(self):
        with self._hazard_lock:
            if self._hazard_started is None:
                return {}
            position = self.hazard_position()
            hazard = self._hazard_status(position)
            return {**{key: hazard[key] for key in ("radius_m", "edge_distance_m", "inside")},
                    "eta_hours": hazard["eta_s"] / 3600,
                    **nearest_safe_point(position, self.hazard_policy)}
