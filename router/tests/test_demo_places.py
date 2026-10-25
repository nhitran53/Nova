"""Demo places (base 2, comms tower), the numbered-suffix guard, and the 'how far' distance line.

Covers router/places.py + places.json, FixtureDatasource demo_places, tools.PLACE_DISTANCES_M, turn.py distance_templates,
the demo_live fake-router keywords and the /pos command. No network, no model, no LLM."""
import copy

import pytest
from test_datasource import FakeRouter, fake_postgres, row

from router.datasource import FixtureDatasource, load_demo_places, load_data_source_policy
from router.demo_live import KeywordRouter, apply_pos, frozen_clock
from router.places import load_gazetteer, resolve_destination, resolve_place
from router.tools import default_registry
from router.turn import POLICY_PATH, TurnHandler, load_policy
from router.verifier import verify_numbers


@pytest.mark.parametrize("text, place_id", [
    ("how far is base 2", "base_2"), ("base2", "base_2"), ("second base", "base_2"), ("take me to base 2", "base_2"),
    ("go to base b", "base_2"), ("take me home", "homebase"), ("go to base", "homebase"), ("how far is the base", "homebase"),
    ("how far is the tower", "comms_tower"), ("tower", "comms_tower"), ("comms tower", "comms_tower"),
])
def test_resolvers_pick_the_expected_place(text, place_id):
    assert resolve_place(text)["place_id"] == place_id
    assert resolve_destination(text)["place"]["place_id"] == place_id


@pytest.mark.parametrize("text", ["base two", "how far is base two", "base 3", "take me to base 3", "go to base ten"])
def test_numbered_suffix_guard_never_falls_back_to_homebase(text):
    assert resolve_place(text) is None
    assert resolve_destination(text)["status"] == "none"


def test_gazetteer_has_no_alias_collision_and_tower_belongs_to_comms_tower():
    gazetteer = load_gazetteer()
    assert "tower" in gazetteer["comms_tower"] and "base 2" in gazetteer["base_2"] and "base" in gazetteer["homebase"]


def test_fixture_knows_the_demo_places_and_the_seed_rows_are_unchanged():
    ds = FixtureDatasource()
    assert ds.place("base_2") == {"code": "BASE2", "name": "Base 2", "distance_m": 1000}
    assert ds.place("comms_tower") == {"code": "TOWER", "name": "Comms Tower", "distance_m": 500}
    assert ds.place("rover")["distance_m"] == 126 and set(ds.locations) >= {"HAB", "ROVER", "BASE2", "TOWER"}


@pytest.mark.parametrize("change", [
    {"locations": [{"code": "HAB", "name": "x", "x_m": 1, "z_m": 1}], "place_aliases": {}},
    {"locations": [], "place_aliases": {"rover": "ROVER"}},
    {"locations": [], "place_aliases": {"nowhere": "NOPE"}},
])
def test_demo_places_collisions_raise(change):
    with pytest.raises(ValueError, match="demo_places"):
        FixtureDatasource(demo_places=change)


def test_demo_places_do_not_mutate_the_shared_policy():
    shared = load_data_source_policy()
    before = copy.deepcopy(shared)
    FixtureDatasource(policy=shared)
    assert shared == before


def test_postgres_fails_closed_for_demo_places():
    ds = fake_postgres([row(30.0), row(0.0)])[0]
    assert ds.place("base_2") is None and ds.place("comms_tower") is None


def test_sim_state_registry_handles_base_2():
    data = default_registry().call("navigation", {"place": "base_2"})["data"]
    assert (data["distance_m"], data["eta_min"], data["place_name"]) == (1000, 17, "base 2")


def test_policy_sections_are_placeholder_and_the_diff_only_adds_keys():
    policy = load_policy()
    assert policy["demo_places"]["status"].startswith("PLACEHOLDER") and policy["distance_templates"]["status"].startswith("PLACEHOLDER")
    assert load_demo_places()["place_aliases"] == {"base_2": "BASE2", "comms_tower": "TOWER"}
    assert policy["templates"]["navigation"] == "Heading to {place_name}. About {distance_m:.0f} meters. Estimated time in minutes: {eta_min:.0f}."


def handler(label="call_navigation"):
    ds = FixtureDatasource(clock=frozen_clock())
    return TurnHandler(FakeRouter(label, 0.97), ds.snapshot, default_registry(datasource=ds)), ds


def test_how_far_speaks_the_distance_line_from_the_tool_result_without_the_llm():
    out = handler()[0].handle("how far is base 2")
    assert out["tool_call"] == {"name": "navigation", "args": {"place": "base_2"}} and out["llm_called"] is False
    assert out["spoken"].startswith("base 2 is 17 minutes away.") and "Heading to" not in out["spoken"]
    assert "oxygen" in out["spoken"] and out["blocked_text"] is None
    assert out["verifier"]["ok"] and verify_numbers(out["spoken"], out["tool_result"])["ok"]


def test_take_me_to_reports_guidance_without_implying_movement():
    out = handler()[0].handle("take me to base 2")
    assert out["spoken"].startswith("base 2 is 17 minutes away.")


@pytest.mark.parametrize("text", ["what is the distance to the tower", "distance from here to the tower", "HOW FAR is the tower?"])
def test_distance_phrases_pick_the_distance_template(text):
    assert handler()[0].handle(text)["spoken"].startswith("comms tower is 9 minutes away.")


def test_how_far_base_two_asks_where_instead_of_navigating():
    out = handler()[0].handle("how far is base two")
    assert out["tool_call"] is None and out["spoken"] == "Where would you like to go?"


def test_keyword_router_sends_how_far_is_to_navigation_but_not_how_far_away():
    router = KeywordRouter()
    assert router.predict("how far is base 2")["label"] == "call_navigation"
    assert router.predict("how far to the tower")["label"] == "call_navigation"
    assert router.predict("distance to the rover")["label"] == "call_navigation"
    assert router.predict("how far away is the moon")["label"] == "call_llm"


def test_pos_command_moves_the_astronaut_and_changes_the_distance():
    turn, ds = handler()
    assert turn.handle("how far is home")["tool_result"]["data"]["distance_m"] == 0
    assert apply_pos(ds, "500 0") == "astronaut moved to x=500 z=0"
    out = turn.handle("how far is home")
    assert out["tool_result"]["data"]["distance_m"] == 500 and out["spoken"].startswith("homebase is 9 minutes away")


@pytest.mark.parametrize("arguments", ["", "5", "a b", "1 2 3", "nan 0", "inf 0"])
def test_pos_rejects_bad_arguments_and_leaves_the_position(arguments):
    ds = FixtureDatasource()
    assert apply_pos(ds, arguments).startswith("usage: /pos") and len(ds.positions) == 1
