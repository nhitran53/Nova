"""Demo "what is this" scene: a whole-utterance phrase answered with a fixed line built from the rover-diagnostics fixture
(snapshot["rover_faults"]); no router, no LLM, no retrieval for speech. Honest wording: diagnostics plus gaze, never vision."""
from pathlib import Path

from .fast_replies import canonical

DEFAULT_DOCS_DIR = Path(__file__).resolve().parent / "corpus" / "sawppy"
ENV_FLAG = "NOVA_DEMO_SCENE"
LINE_KEYS = ("scene_line", "no_fault_line", "queued_line", "declined_line")
PRINTER_SOURCE = {"printer": "3D printer"}  # makes the spoken "3D" traceable for the number verifier


def validate_demo_scene(section, where="turn_policy.json"):
    """Raise ValueError for a demo_scene section the code cannot trust. Lines pass through _speak: only the "3D" of "3D printer" may be a digit."""
    if not isinstance(section, dict):
        raise ValueError(f"{where}: 'demo_scene' must be an object")
    if not isinstance(section.get("enabled"), bool):
        raise ValueError(f"{where}: demo_scene.enabled must be a boolean")
    for key in ("phrases", "wake_prefixes"):
        values = section.get(key)
        if not isinstance(values, list) or (key == "phrases" and not values) or not all(isinstance(v, str) and canonical(v) for v in values):
            raise ValueError(f"{where}: demo_scene.{key} must be a list of non-empty strings")
    for key in LINE_KEYS:
        line = section.get(key)
        if not isinstance(line, str) or not line.strip() or any(c.isdigit() for c in line.replace("3D", "")):
            raise ValueError(f"{where}: demo_scene.{key} must be a non-empty string without digits (except 3D)")
    if "{part_label}" not in section["scene_line"] or "{print_part}" not in section["scene_line"]:
        raise ValueError(f"{where}: demo_scene.scene_line must contain {{part_label}} and {{print_part}}")


def demo_scene_enabled(policy, options=None, environ=None):
    """The policy flag, OR llm_options["demo_scene"], OR NOVA_DEMO_SCENE=1 in `environ`."""
    section = policy.get("demo_scene") or {}
    return bool(section.get("enabled") or (options or {}).get("demo_scene") or (environ or {}).get(ENV_FLAG) == "1")


def match_demo_scene(text, policy):
    """True when the WHOLE utterance is a scene phrase, as spoken or after one leading wake prefix; False when disabled."""
    section = policy.get("demo_scene")
    if not section or not section["enabled"]:
        return False
    phrases = {canonical(p) for p in section["phrases"]}
    spoken = canonical(text)
    if spoken in phrases:
        return True
    return any(spoken.startswith(prefix + " ") and spoken[len(prefix) + 1:] in phrases
               for prefix in sorted((canonical(p) for p in section["wake_prefixes"]), key=len, reverse=True))


def printable_fault(snapshot):
    """First damaged rover fault that names a printable part, else None. Malformed fixture entries are ignored."""
    for fault in snapshot.get("rover_faults") or ():
        if (isinstance(fault, dict) and fault.get("status") == "damaged" and isinstance(fault.get("print_part"), str)
                and isinstance(fault.get("label"), str) and isinstance(fault.get("part"), str)):
            return fault
    return None
