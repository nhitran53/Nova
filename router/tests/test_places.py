"""Unit tests for the deterministic place resolver (router/places.py)."""
import json

import pytest

from router.places import DEFAULT_THRESHOLD, load_gazetteer, normalise, resolve_destination, resolve_place


@pytest.mark.parametrize("text, place_id", [
    ("take me home", "homebase"),
    ("take me to home", "homebase"),
    ("directions to homebase", "homebase"),
    ("back to the base", "homebase"),
    ("take me back to the base please", "homebase"),
    ("lead me to the home base", "homebase"),
    ("Take me to Home!", "homebase"),
    ("directions to homebse", "homebase"),
    ("go to home bse", "homebase"),
    ("which way is the airlok", "airlock"),
    ("zoom in on the air lock", "airlock"),
    ("how far is the ridge", "ridge"),
    ("head to the landing pad", "landing_pad"),
    ("show the crater rim on the map", "crater_rim"),
    ("where's the rover", "rover"),
])
def test_resolves_variants_and_misspellings(text, place_id):
    result = resolve_place(text)
    assert result is not None and result["place_id"] == place_id
    assert result["score"] >= DEFAULT_THRESHOLD


def test_result_shape_and_exact_score():
    result = resolve_place("take me to the home base")
    assert result == {"place_id": "homebase", "matched_text": "home base", "score": 1.0}


def test_home_base_and_homebase_are_the_same_place():
    assert resolve_place("home base")["place_id"] == resolve_place("homebase")["place_id"] == "homebase"


@pytest.mark.parametrize("text", [
    "", "   ", "to the", "to the my a", "what time is it", "hello nova", "zoom in please",
    "take me over there", "go over to it",  # "over" must not fuzzy-match "rover"
    "that is a bridge",  # first letter differs from "ridge"
    "asdf qwerty zxcv",
])
def test_no_match_returns_none(text):
    assert resolve_place(text) is None


def test_stopwords_alone_never_match_even_with_low_threshold():
    assert resolve_place("to the", threshold=0.0) is None
    assert resolve_place("my a of", threshold=0.0) is None


def test_threshold_is_configurable():
    assert resolve_place("directions to homebse", threshold=0.99) is None
    assert resolve_place("directions to homebse", threshold=0.9)["place_id"] == "homebase"


def test_exact_match_wins_over_fuzzy_match_elsewhere_in_utterance():
    result = resolve_place("the airlok or the rover", threshold=0.8)
    assert result["place_id"] == "rover" and result["score"] == 1.0


@pytest.fixture
def two_place_gazetteer(tmp_path):
    path = tmp_path / "places.json"
    path.write_text(json.dumps({"places": {"alpha": ["tower"], "beta": ["bunker"]}}))
    return load_gazetteer(path)


def test_equal_scores_prefer_longer_window_then_earlier_then_id(two_place_gazetteer):
    g = two_place_gazetteer
    assert resolve_place("tower then bunker", gazetteer=g)["place_id"] == "alpha"  # earlier
    assert resolve_place("bunker then tower", gazetteer=g)["place_id"] == "beta"
    g2 = {"zeta": ["north gate"], "alpha": ["gate"]}
    assert resolve_place("the north gate", gazetteer=g2)["place_id"] == "zeta"  # longer window wins


def test_equal_scores_same_position_break_by_place_id():
    g = {"zeta": ["gate"], "alpha": ["gate"]}  # built by hand: load_gazetteer would reject the shared alias
    assert resolve_place("the gate", gazetteer=g)["place_id"] == "alpha"


def test_load_gazetteer_rejects_shared_alias_and_stopword_alias(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"places": {"alpha": ["gate"], "beta": ["gate"]}}))
    with pytest.raises(ValueError, match="belongs to both"):
        load_gazetteer(path)
    path.write_text(json.dumps({"places": {"alpha": ["the gate"]}}))
    with pytest.raises(ValueError, match="stopword"):
        load_gazetteer(path)


def test_shipped_gazetteer_loads_and_is_normalised():
    g = load_gazetteer()
    assert "homebase" in g and "home" in g["homebase"]
    assert all(alias == normalise(alias) for aliases in g.values() for alias in aliases)


def test_exported_from_package():
    import router

    assert router.resolve_place is resolve_place


# ---- destination-aware resolution (review finding 2) ----
def ids(entries):
    return [e["place_id"] for e in entries]


@pytest.mark.parametrize("text, place_id", [
    ("take me home", "homebase"),
    ("take me to home", "homebase"),
    ("directions to homebase", "homebase"),
    ("homebse", "homebase"),
    ("go from the rover to the airlock", "airlock"),  # origin first, destination second
    ("go to the airlock from the rover", "airlock"),  # destination first, origin second
    ("take me from the rover to the airlock", "airlock"),
    ("take me back to the base from the crater", "homebase"),
    ("from the habitat take me home", "homebase"),  # destination without a marker
    ("head toward the ridge away from the rover", "ridge"),
    ("zoom in on the crater rim", "crater_rim"),  # "crater rim" is one place, not crater + rim
    ("take me to the home base", "homebase"),  # "home base" is one place, not home + base
])
def test_destination_picks_destination_not_origin(text, place_id):
    result = resolve_destination(text)
    assert result["status"] == "resolved" and result["place"]["place_id"] == place_id
    assert ids(result["candidates"]) == [place_id]


def test_origin_is_reported_as_excluded_with_its_role():
    result = resolve_destination("go from the rover to the airlock")
    assert [(e["place_id"], e["role"]) for e in result["excluded"]] == [("rover", "origin")]


@pytest.mark.parametrize("text, place_id, negated", [
    ("take me to the rover not home", "rover", "homebase"),
    ("take me to the rover, not to home", "rover", "homebase"),
    ("not home, take me to the rover", "rover", "homebase"),
    ("go to the rover instead of the airlock", "rover", "airlock"),
    ("go to the rover rather than the airlock", "rover", "airlock"),
    ("take me to the airlock never the habitat", "airlock", "habitat"),
    ("take me to the rover don't go to the airlock", "rover", "airlock"),
    ("take me to the rover dont go home", "rover", "homebase"),
    ("go to the airlock and avoid the crater", "airlock", "crater"),
])
def test_negated_places_are_excluded(text, place_id, negated):
    result = resolve_destination(text)
    assert result["status"] == "resolved" and result["place"]["place_id"] == place_id
    assert [(e["place_id"], e["role"]) for e in result["excluded"]] == [(negated, "negated")]


@pytest.mark.parametrize("text, expected", [
    ("go to the rover or the airlock", ["rover", "airlock"]),
    ("take me to the airlock or the rover", ["airlock", "rover"]),  # order of mention is not a tie-break
    ("zoom in on the airlock at home", ["airlock", "homebase"]),
    ("navigate to the ridge and the crater", ["ridge", "crater"]),
])
def test_multiple_unresolved_places_are_ambiguous_and_never_picked_by_position(text, expected):
    result = resolve_destination(text)
    assert result["status"] == "ambiguous" and result["place"] is None
    assert ids(result["candidates"]) == expected


def test_one_place_named_twice_is_not_ambiguous():
    result = resolve_destination("take me to home or the base")
    assert result["status"] == "resolved" and result["place"]["place_id"] == "homebase"


@pytest.mark.parametrize("text", ["", "what time is it", "take me from the rover", "not home", "take me to the creator"])
def test_no_destination_returns_none(text):
    result = resolve_destination(text)
    assert result["status"] == "none" and result["place"] is None and result["candidates"] == []


def test_creator_is_not_matched_to_crater():
    assert resolve_place("take me to the creator") is None
    assert resolve_destination("take me to the creator")["status"] == "none"


def test_resolve_destination_exported_from_package():
    import router

    assert router.resolve_destination is resolve_destination


@pytest.mark.parametrize("text", [
    "go from an unknown location to home",
    "from the remote unknown starting position toward home",
    "go from the rover staging area back to home",
])
def test_destination_spans_are_extracted_before_matching_origins(text):
    result = resolve_destination(text)
    assert result["status"] == "resolved"
    assert result["place"]["place_id"] == "homebase"


@pytest.mark.parametrize("text, place_id", [
    ("no, take me to the airlock", "airlock"),
    ("no go to the airlock", "airlock"),
    ("rather go to the airlock", "airlock"),
    ("instead take me to the airlock", "airlock"),
    ("do not go to the crater, go to the airlock", "airlock"),
    ("never go to the crater but go to the airlock", "airlock"),
])
def test_only_dont_not_never_carry_negation_over_a_command_verb(text, place_id):
    result = resolve_destination(text)
    assert result["status"] == "resolved" and result["place"]["place_id"] == place_id


def test_negation_scope_does_not_expire_after_three_tokens():
    result = resolve_destination("go to the rover not to the distant old home")
    assert result["status"] == "resolved"
    assert result["place"]["place_id"] == "rover"
    assert [(e["place_id"], e["role"]) for e in result["excluded"]] == [("homebase", "negated")]


def test_avoid_and_dont_alone_leave_no_destination():
    for text in ("avoid the crater", "don't take me to the crater", "do not go to the crater", "never go to the crater",
                 "do not take me to the crater", "please never navigate to the crater"):
        result = resolve_destination(text)
        assert result["status"] == "none" and result["place"] is None
        assert [(e["place_id"], e["role"]) for e in result["excluded"]] == [("crater", "negated")]


@pytest.mark.parametrize("text", ["how far am I from tower", "how far am i from the tower", "distance from tower"])
def test_distance_cue_with_only_an_origin_place_resolves_that_place(text):
    result = resolve_destination(text)
    assert result["status"] == "resolved" and result["place"]["place_id"] == "comms_tower"
    assert ids(result["candidates"]) == ["comms_tower"] and result["excluded"] == []


def test_distance_cue_does_not_change_an_explicit_origin_and_destination():
    result = resolve_destination("go from the rover to the airlock")
    assert result["place"]["place_id"] == "airlock"
    assert [(e["place_id"], e["role"]) for e in result["excluded"]] == [("rover", "origin")]


@pytest.mark.parametrize("text", ["how far from tower, not home", "distance from tower not the airlock",
                                  "how far is it, don't go to the tower", "go from tower"])
def test_distance_cue_with_a_negated_place_or_no_cue_still_gives_none(text):
    assert resolve_destination(text)["status"] == "none"
