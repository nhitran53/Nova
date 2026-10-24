"""Coverage / leakage checks for data/gen_synthetic.py output and the hand-written paraphrase_test.jsonl."""
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from router import load_labels

HERE = Path(__file__).parent.parent
LABELS = load_labels(HERE / "labels.json")
_spec = importlib.util.spec_from_file_location("gen_synthetic", HERE / "data" / "gen_synthetic.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def rows(name):
    path = HERE / "data" / f"{name}.jsonl"
    if not path.is_file():
        pytest.skip("run data/gen_synthetic.py first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_paraphrase_set_is_balanced_valid_and_hand_written_size():
    para = rows("paraphrase_test")
    counts = Counter(r["label"] for r in para)
    assert set(counts) == set(LABELS) and len(set(counts.values())) == 1  # balanced
    assert 56 <= len(para) <= 72
    texts = {r["text"] for r in para}
    assert {"take me home", "take me to home"} <= texts


def test_paraphrase_and_demo_texts_never_appear_in_any_generated_split():
    held = gen.held_out_texts()
    assert len(held) >= 100
    for split in ("train", "dev", "test"):
        leaked = [r["text"] for r in rows(split) if gen.normalise(r["text"]) in held]
        assert not leaked, (split, leaked[:5])


def test_generator_excludes_held_out_texts():
    banned = {"take me home"}
    data = gen.generate(["call_navigation"], seed=0, sizes={"train": 60, "dev": 5, "test": 5}, banned=banned)
    assert all(gen.normalise(r["text"]) not in banned for r in data["train"])


def home_nav(pattern):
    return sum(bool(re.search(pattern, r["text"].lower())) for r in rows("train") if r["label"] == "call_navigation")


def test_navigation_train_has_home_with_and_without_to():
    assert home_nav(r"\b(take|bring|get|lead|guide|walk|escort|direct|steer) me home\b") >= 8
    assert home_nav(r"\b(to|toward) (my |our )?home\b") >= 5
    assert home_nav(r"\bhomebase\b|\bhome base\b") >= 10


def test_place_words_occur_in_many_classes_so_they_do_not_decide_the_label():
    for word in ("home", "base"):
        labels = {r["label"] for r in rows("train") if re.search(rf"\b{word}\b", r["text"].lower())}
        assert len(labels) >= 5, (word, labels)
    assert "OOD" in {r["label"] for r in rows("train") if re.search(r"\bhome\b", r["text"].lower())}
    assert "NO_TOOL" in {r["label"] for r in rows("train") if re.search(r"\bhome\b", r["text"].lower())}


def test_every_label_has_many_distinct_train_templates():
    for label in LABELS:
        assert len(gen.TEMPLATES[label]["train"]) >= 20, label
