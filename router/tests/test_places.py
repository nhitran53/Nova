"""Unit tests for the deterministic place resolver (router/places.py)."""
import json

import pytest

from router.places import DEFAULT_THRESHOLD, load_gazetteer, normalise, resolve_place


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
