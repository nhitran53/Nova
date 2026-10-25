"""Fresh Unity locations with short replies. Simulator positions are never used here."""
import math
import re

UNAVAILABLE = "I can't read your current Unity location. Check that the scene is running and connected."


def match_return_to_start(text):
    text = re.sub(r"[^a-z0-9 ]", "", text.lower())
    text = re.sub(r"^(?:hey )?nova\s+", "", text).strip()
    return bool(re.fullmatch(
        r"(?:please )?(?:return(?: me)?|take me back|bring me back|go back) to (?:the )?"
        r"(?:start|starting (?:position|point)|rover inspection (?:position|point)|damaged wheel)(?: please)?", text))


def match_location(text):
    text = re.sub(r"[^a-z0-9 ]", "", text.lower().replace("'", ""))
    text = re.sub(r"^(?:hey )?nova\s+", "", text).strip()
    patterns = (
        r"where (?:am i|are we)(?: (?:now|on mars|right now))?",
        r"(?:whats|what is) (?:near(?:by)?|around) (?:me|us|here)",
        r"(?:whats|what is) nearby",
        r"what (?:landmarks|things) are (?:near(?:by)?|around)(?: (?:me|us|here))?",
        r"(?:whats|what is) my (?:current )?location(?: on mars)?",
        r"what are my (?:current |mars )?coordinates",
        r"what region(?: of mars)? am i (?:in|exploring)",
        r"how far (?:am i from|is) (?:the )?(?:shuttle|rover)(?: from me)?",
        r"where is (?:the )?(?:shuttle|rover|rover wheel)(?: relative to me)?",
    )
    return any(re.fullmatch(pattern, text) for pattern in patterns)


def validate_snapshot(payload):
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or type(payload.get("seq")) is not int:
        raise ValueError("Invalid spatial snapshot header")
    if type(payload.get("available")) is not bool:
        raise ValueError("Missing availability")
    if not payload["available"]:
        return {"available": False, "error": "Unity location is unavailable", "latitude": None, "longitude": None,
                "coordinates_verified": False, "nearby_landmarks": []}

    def number(value):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("Invalid spatial number")
        return value

    def vector(value):
        if not isinstance(value, dict):
            raise ValueError("Invalid spatial vector")
        return {axis: number(value.get(axis)) for axis in ("x", "y", "z")}

    result = {"available": True, "coordinates_verified": payload.get("coordinates_verified") is True,
              "world_position": vector(payload.get("world_position")),
              "head_position": vector(payload.get("head_position")), "head_forward": vector(payload.get("head_forward")),
              "world_north": vector(payload.get("world_north")), "latitude": None, "longitude": None,
              "environment": None, "region": None, "coordinate_reference": None, "nearby_landmarks": [],
              "navigation": payload.get("navigation") or {}}
    if result["coordinates_verified"]:
        lat, lon = number(payload.get("latitude")), number(payload.get("longitude"))
        if not -90 <= lat <= 90 or not 0 <= lon < 360:
            raise ValueError("Coordinates outside source convention")
        result.update(latitude=lat, longitude=lon, environment=payload.get("environment"), region=payload.get("region"),
                      coordinate_reference=payload.get("coordinate_reference"))
    landmarks = payload.get("nearby_landmarks", [])
    if not isinstance(landmarks, list) or len(landmarks) > 32:
        raise ValueError("Invalid landmark list")
    for item in landmarks:
        if not isinstance(item, dict) or not all(isinstance(item.get(k), str) and 0 < len(item[k]) <= 200 for k in ("id", "name")):
            raise ValueError("Invalid landmark identity")
        distance, bearing = number(item.get("distance_m")), number(item.get("bearing_deg"))
        if distance < 0 or not 0 <= bearing < 360:
            raise ValueError("Invalid landmark measurement")
        if item.get("direction") not in ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"):
            raise ValueError("Invalid compass direction")
        if item.get("relative_direction") not in ("in front of you", "behind you", "to your right", "to your left", "unavailable"):
            raise ValueError("Invalid relative direction")
        result["nearby_landmarks"].append({**item, "position": vector(item.get("position"))})
    return result


def describe_location(data, question):
    if not data.get("available"):
        return UNAVAILABLE
    question = question.lower()
    if "coordinate" in question:
        if not data["coordinates_verified"]:
            return "Precise Mars coordinates are unavailable because terrain georeferencing could not be verified."
        return (f"Your source spherical latitude is {data['latitude']:.5f} degrees. "
                f"Your east longitude is {data['longitude']:.5f} degrees.")
    landmarks = data["nearby_landmarks"]
    if re.search(r"\b(?:nearby|near|around)\b", question) and not re.search(r"\b(?:shuttle|rover|wheel)\b", question):
        if not landmarks:
            return "No registered landmarks are nearby."
        closest = sorted(landmarks, key=lambda p: p["distance_m"])[:3]
        return " ".join(f"{p['name']} is about {p['distance_m']:.0f} meters {p['direction']}." for p in closest)
    target = "shuttle" if "shuttle" in question else "rover_front_right_wheel" if "wheel" in question else "rover" if "rover" in question else None
    if target:
        landmark = next((p for p in landmarks if p["id"] == target or (target == "shuttle" and p["id"] == "shuttle_001")), None)
        if landmark is None:
            return "That landmark isn't registered in the current Unity scene."
        relative = "" if landmark["relative_direction"] == "unavailable" else f", {landmark['relative_direction']}"
        return (f"{landmark['name']} is about {landmark['distance_m']:.0f} meters {landmark['direction']}{relative}. "
                "That's straight-line ground distance.")
    # General location questions report the verified region.
    who = "We're" if re.search(r"\b(?:we|us)\b", question) else "You're"
    if data.get("region") and data.get("environment"):
        return f"{who} in {data['region']} on {data['environment']}."
    return f"{who} on Mars, but the exact region could not be verified."


def make_location_tool(bridge):
    def locate(question):
        try:
            data = bridge.request_spatial()
        except (TimeoutError, ValueError):
            return {"tool": "get_current_location", "ok": False, "data": {"line": UNAVAILABLE}}
        return {"tool": "get_current_location", "ok": data.get("available", False), "data": data}
    return locate
