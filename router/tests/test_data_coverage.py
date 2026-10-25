"""Coverage / leakage checks for data/gen_synthetic.py output and the hand-written paraphrase_test.jsonl."""
import importlib.util
import itertools
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from router import load_labels

HERE = Path(__file__).parent.parent
LABELS = load_labels(HERE / "labels.json")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / "data" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load("gen_synthetic")  # v2 generator
gen_v1 = _load("gen_synthetic_v1")  # frozen generator of the v1 training data
DATA_DIRS = {"v1": HERE / "data", "v2": HERE / "data" / "v2"}


def rows(name, version="v1"):
    path = DATA_DIRS[version] / f"{name}.jsonl" if name != "paraphrase_test" else HERE / "data" / "paraphrase_test.jsonl"
    if not path.is_file():
        pytest.skip(f"run data/gen_synthetic{'_v1' if version == 'v1' else ''}.py first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_paraphrase_set_is_balanced_valid_and_hand_written_size():
    para = rows("paraphrase_test")
    counts = Counter(r["label"] for r in para)
    assert set(counts) == set(LABELS) and counts["llm"] == 3 * counts["call_safety"]  # three legacy LLM intents merged
    assert 56 <= len(para) <= 72
    texts = {r["text"] for r in para}
    assert {"take me home", "take me to home"} <= texts


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_paraphrase_and_demo_texts_never_appear_in_any_generated_split(version):
    held = gen.held_out_texts()
    assert len(held) >= 100
    for split in ("train", "dev", "test"):
        leaked = [r["text"] for r in rows(split, version) if gen.normalise(r["text"]) in held]
        assert not leaked, (version, split, leaked[:5])


def test_generator_excludes_held_out_texts():
    banned = {"take me home"}
    data = gen.generate(["call_navigation"], seed=0, sizes={"train": 60, "dev": 5, "test": 5}, banned=banned)
    assert all(gen.normalise(r["text"]) not in banned for r in data["train"])


def home_nav(pattern, version):
    return sum(bool(re.search(pattern, r["text"].lower())) for r in rows("train", version)
               if r["label"] == "call_navigation")


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_navigation_train_has_home_with_and_without_to(version):
    assert home_nav(r"\b(take|bring|get|lead|guide|walk|escort|direct|steer) me home\b", version) >= 8
    assert home_nav(r"\b(to|toward) (my |our )?home\b", version) >= 5
    assert home_nav(r"\bhomebase\b|\bhome base\b", version) >= 10


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_place_words_occur_in_many_classes_so_they_do_not_decide_the_label(version):
    for word in ("home", "base"):
        labels = {r["label"] for r in rows("train", version) if re.search(rf"\b{word}\b", r["text"].lower())}
        assert len(labels) >= 5, (word, labels)
    assert "OOD" in {r["fine_label"] for r in rows("train", version) if re.search(r"\bhome\b", r["text"].lower())}
    assert "NO_TOOL" in {r["fine_label"] for r in rows("train", version) if re.search(r"\bhome\b", r["text"].lower())}


@pytest.mark.parametrize("module", [gen, gen_v1])
def test_every_label_has_many_distinct_train_templates(module):
    for fine_label in module.TEMPLATES:
        assert len(module.TEMPLATES[fine_label]["train"]) >= 20, fine_label


# ---- v1 stays reproducible ----
def test_v1_generator_reproduces_committed_v1_data_byte_for_byte(tmp_path):
    gen_v1.main(["--out-dir", str(tmp_path)])
    for split in ("train", "dev", "test"):
        assert (tmp_path / f"{split}.jsonl").read_bytes() == (DATA_DIRS["v1"] / f"{split}.jsonl").read_bytes(), split


def test_committed_v2_data_matches_v2_generator():
    generated = gen.generate(LABELS, seed=0)
    for split in ("train", "dev", "test"):
        assert rows(split, "v2") == generated[split], split


# ---- v2: no shared frames or slot-fill families across splits (review finding 3) ----
SPLITS = ("train", "dev", "test")
PAIRS = list(itertools.combinations(SPLITS, 2))


@pytest.mark.parametrize("a, b", PAIRS)
def test_v2_expanded_frames_are_disjoint_across_splits(a, b):
    # frames made only of slots (e.g. "{} {}", gibberish) carry no wording; their fills are checked via SYLLABLES
    frames = {s: {f for f in set().union(*gen.expanded_frames(s).values()) if re.search(r"[a-z]", f)} for s in SPLITS}
    assert len(frames[a]) > 100
    assert not frames[a] & frames[b], sorted(frames[a] & frames[b])[:10]


@pytest.mark.parametrize("a, b", PAIRS)
def test_v2_family_slot_values_are_disjoint_across_splits(a, b):
    fa, fb = gen.family_values(a), gen.family_values(b)
    assert set(fa) == set(fb) == {*gen.FAMILY_SLOTS, "dest_form"}
    for family in fa:
        assert fa[family] and fb[family], family
        assert not {v.lower() for v in fa[family]} & {v.lower() for v in fb[family]}, (family, a, b)


@pytest.mark.parametrize("a, b", PAIRS)
def test_v2_content_slots_noise_and_extra_places_are_disjoint_across_splits(a, b):
    for slot in gen.POOLS:
        assert not set(gen.pool(slot, a)) & set(gen.pool(slot, b)), (slot, a, b)
    for kind, per_split in gen.NOISE.items():
        assert not set(per_split[a]) & set(per_split[b]), (kind, a, b)
    slips = {s: set(gen.STT_SLIPS[s].items()) for s in SPLITS}
    assert not slips[a] & slips[b]
    assert not set(gen.EXTRA_PLACES[a]) & set(gen.EXTRA_PLACES[b])
    assert not set(gen.SYLLABLES[a]) & set(gen.SYLLABLES[b])


def test_v2_only_documented_entity_vocabulary_is_shared_by_all_splits():
    for split in SPLITS:
        assert not set(gen.PLACES) & set(gen.EXTRA_PLACES[split])
    assert gen.WAKE_FORMS == ["nova", "hey nova"]


def test_v2_literal_template_text_never_contains_another_splits_multiword_family_fill():
    templates = {s: [t.lower() for label in gen.TEMPLATES.values() for t in label[s]] for s in SPLITS}
    for split in SPLITS:
        fills = [v.lower() for slot in gen.FAMILY_SLOTS for v in gen.pool(slot, split) if " " in v]
        forms = [f.replace("{P}", "").replace("{H}", "").strip().lower() for f in
                 gen.DEST_FORMS[split] + gen.BARE_HOME_FORMS[split]]
        fills += [f for f in forms if " " in f]
        for other in SPLITS:
            if other == split:
                continue
            leaked = [(fill, t) for fill in fills for t in templates[other] if re.search(rf"\b{re.escape(fill)}\b", t)]
            assert not leaked, (split, other, leaked[:5])


def test_v2_generated_rows_match_split_sizes_and_unique_texts():
    texts = []
    for split, n in gen.SPLIT_SIZES.items():
        got = rows(split, "v2")
        counts = Counter(r["label"] for r in got)
        assert set(counts) == set(LABELS)
        assert counts == {label: n * (3 if label == "llm" else 1) for label in LABELS}
        texts += [r["text"] for r in got]
    assert len(texts) == len(set(texts))


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("split", ["train", "dev", "test", "paraphrase_test"])
def test_six_class_labels_are_rebuilt_from_preserved_fine_label(version, split):
    for row in rows(split, version):
        fine = row["fine_label"]
        assert row["label"] == ("llm" if fine in {"call_llm", "NO_TOOL", "OOD"} else fine)
        assert row["label"] in LABELS
