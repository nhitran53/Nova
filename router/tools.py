"""Tool registry with fixed schemas, plus FAKE tools over a fabricated sim state (no real tool integration yet).

Hard rules enforced here: at most MAX_TOOLS (8) registered tools, and at most ONE tool call per turn
(`begin_turn()` resets the counter; a second `call()` in the same turn raises OneCallPerTurnError).
Schemas are fixed at registration: `{"required": [...], "properties": {arg: type}}`. `call()` rejects unknown tools,
unknown or missing args and wrong types before the tool runs.

Tool results are `{"tool": name, "ok": bool, "data": {...}}`; every number the assistant speaks must come from `data`.
All sim numbers below are fabricated demo values, not real suit telemetry.
"""
MAX_TOOLS = 8
PLACE_DISTANCES_M = {  # fabricated: (distance in meters, walking minutes)
    "homebase": (120, 2), "habitat": (150, 3), "airlock": (80, 1), "rover": (200, 4), "lander": (900, 15),
    "landing_pad": (850, 14), "landing_site": (1200, 20), "crater": (600, 10), "crater_rim": (650, 11),
    "ridge": (500, 9), "outpost": (1500, 25), "greenhouse": (140, 3), "supply_depot": (300, 5),
    "sample_depot": (350, 6), "comms_tower": (400, 7), "solar_farm": (700, 12), "ice_shelf": (2000, 33),
    "lava_tube": (1800, 30), "dune_field": (1100, 18), "quarry": (950, 16), "east_outcrop": (1300, 22),
}


class ToolRegistryError(Exception):
    pass


class ToolLimitError(ToolRegistryError):
    """More than MAX_TOOLS tools registered."""


class OneCallPerTurnError(ToolRegistryError):
    """A second tool call was attempted within one turn."""


class SchemaError(ToolRegistryError):
    """Unknown tool, or call arguments that do not match the tool's fixed schema."""


class ToolRegistry:
    def __init__(self, max_tools=MAX_TOOLS):
        self.max_tools = max_tools
        self._tools = {}
        self.calls_this_turn = 0

    def register(self, name, schema, fn):
        if name in self._tools:
            raise ToolRegistryError(f"tool {name!r} already registered")
        if len(self._tools) >= self.max_tools:
            raise ToolLimitError(f"cannot register {name!r}: registry already holds {self.max_tools} tools")
        self._tools[name] = (schema, fn)

    def names(self):
        return list(self._tools)

    def schema(self, name):
        if name not in self._tools:
            raise SchemaError(f"unknown tool {name!r}")
        return self._tools[name][0]

    def begin_turn(self):
        self.calls_this_turn = 0

    def call(self, name, args):
        schema = self.schema(name)
        if self.calls_this_turn >= 1:
            raise OneCallPerTurnError(f"second tool call {name!r} in one turn")
        properties = schema["properties"]
        unknown = set(args) - set(properties)
        missing = [arg for arg in schema["required"] if arg not in args]
        if unknown or missing:
            raise SchemaError(f"{name}: unknown args {sorted(unknown)}, missing args {missing}")
        for arg, value in args.items():
            if not isinstance(value, properties[arg]):
                raise SchemaError(f"{name}.{arg}: expected {properties[arg].__name__}, got {type(value).__name__}")
        self.calls_this_turn += 1
        return self._tools[name][1](**args)


class SimState:
    """Fabricated suit/vitals readings. `snapshot()` doubles as the alert source; tests and the demo mutate fields."""

    def __init__(self, **overrides):
        self.values = {"o2_pct": 87.5, "minutes_remaining": 212, "bpm": 72, "radiation_usv_h": 0.34, "temp_c": 21.5,
                       "suit_pressure_kpa": 29.6}
        self.values.update(overrides)

    def snapshot(self):
        return dict(self.values)


def default_registry(sim=None):
    """Six fake tools (safety, navigation, map, data, o2, send_message) over `sim`; leaves room for two more."""
    sim = SimState() if sim is None else sim
    registry = ToolRegistry()

    def safety():
        s = sim.snapshot()
        go = s["o2_pct"] >= 16.0 and 20.0 <= s["suit_pressure_kpa"] <= 40.0
        return {"tool": "safety", "ok": True, "data": {"status_word": "go" if go else "no go"}}

    def place_result(tool, place):
        if place not in PLACE_DISTANCES_M:
            return {"tool": tool, "ok": False, "data": {"error": "unknown place"}}
        distance_m, eta_min = PLACE_DISTANCES_M[place]
        return {"tool": tool, "ok": True, "data": {"place": place, "place_name": place.replace("_", " "),
                                                   "distance_m": distance_m, "eta_min": eta_min}}

    def data():
        s = sim.snapshot()
        return {"tool": "data", "ok": True, "data": {k: s[k] for k in ("bpm", "radiation_usv_h", "temp_c")}}

    def o2():
        s = sim.snapshot()
        return {"tool": "o2", "ok": True, "data": {k: s[k] for k in ("o2_pct", "minutes_remaining")}}

    def send_message(recipient, urgency, message):
        """FAKE: nothing is sent anywhere. Returns what a real relay would acknowledge (no numbers)."""
        return {"tool": "send_message", "ok": True,
                "data": {"recipient": recipient, "recipient_name": recipient.replace("_", " "), "urgency": urgency,
                         "status": "queued"}}

    registry.register("safety", {"required": [], "properties": {}}, safety)
    registry.register("navigation", {"required": ["place"], "properties": {"place": str}},
                      lambda place: place_result("navigation", place))
    registry.register("map", {"required": ["place"], "properties": {"place": str}},
                      lambda place: place_result("map", place))
    registry.register("data", {"required": [], "properties": {}}, data)
    registry.register("o2", {"required": [], "properties": {}}, o2)
    registry.register("send_message", {"required": ["recipient", "urgency", "message"],
                                       "properties": {"recipient": str, "urgency": str, "message": str}}, send_message)
    return registry
