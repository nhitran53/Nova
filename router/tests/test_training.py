"""Tests for the synthetic generator, evaluation metrics, temperature fitting and meta-aware Router.load."""
import importlib.util
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from router import Router, load_labels
from router.core import META_FILENAME, base_model_dir
from router.evaluate import (build_report, classification_metrics, expected_calibration_error, load_jsonl,
                             softmax, threshold_sweep)
from router.train import fit_temperature

HERE = Path(__file__).parent.parent
LABELS_PATH = HERE / "labels.json"
MODEL_DIR = Path(os.environ.get("ROUTER_MODEL_DIR", base_model_dir()))

_spec = importlib.util.spec_from_file_location("gen_synthetic", HERE / "data" / "gen_synthetic.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


# ---- generator ----
def test_generator_covers_every_label_with_requested_sizes():
    labels = load_labels(LABELS_PATH)
    data = gen.generate(labels, seed=0, sizes={"train": 20, "dev": 5, "test": 5})
    for split, n in {"train": 20, "dev": 5, "test": 5}.items():
        counts = {label: 0 for label in labels}
        for row in data[split]:
            counts[row["label"]] += 1
        assert counts == {label: n * (3 if label == "llm" else 1) for label in labels}


def test_generator_is_deterministic_and_seed_sensitive():
    labels = load_labels(LABELS_PATH)
    sizes = {"train": 10, "dev": 3, "test": 3}
    assert gen.generate(labels, 1, sizes) == gen.generate(labels, 1, sizes)
    assert gen.generate(labels, 1, sizes) != gen.generate(labels, 2, sizes)


def test_generator_texts_unique_across_splits_and_labels():
    data = gen.generate(load_labels(LABELS_PATH), 0, {"train": 30, "dev": 8, "test": 8})
    texts = [row["text"] for rows in data.values() for row in rows]
    assert len(texts) == len(set(texts))


def test_template_pools_disjoint_across_splits_and_cover_labels():
    assert set(load_labels(LABELS_PATH)) - {"llm"} <= set(gen.TEMPLATES)
    assert {"call_llm", "NO_TOOL", "OOD"} <= set(gen.TEMPLATES)
    for label, pools in gen.TEMPLATES.items():
        train, dev, test = (set(pools[s]) for s in ("train", "dev", "test"))
        assert not (train & dev or train & test or dev & test), label


def test_committed_jsonl_matches_label_file_and_sizes():
    labels = load_labels(LABELS_PATH)
    for split, n in gen.SPLIT_SIZES.items():
        path = HERE / "data" / f"{split}.jsonl"
        if not path.is_file():
            pytest.skip("run data/gen_synthetic.py first")
        _, ys = load_jsonl(path, labels)
        assert np.bincount(ys, minlength=len(labels)).tolist() == [n * (3 if label == "llm" else 1) for label in labels]


def test_load_jsonl_rejects_unknown_label(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({"text": "hi", "label": "nope"}) + "\n")
    with pytest.raises(ValueError, match="unknown label"):
        load_jsonl(path, ["a", "b"])


# ---- metrics ----
def test_classification_metrics_known_confusion():
    m = classification_metrics(np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 0]), 3)
    assert m["confusion"] == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"][:2] == pytest.approx([0.5, 2 / 3])
    assert m["recall"] == pytest.approx([0.5, 1.0, 0.0])
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 0.0) / 3)


def test_ece_zero_when_confidence_matches_accuracy_and_large_when_overconfident():
    y = np.array([0, 0, 0, 0])
    exact = np.array([[0.75, 0.25]] * 4)  # predicts class 0 at 0.75 but see below: all correct -> gap 0.25
    assert expected_calibration_error(exact, y) == pytest.approx(0.25)
    wrong = np.array([[0.99, 0.01]] * 4)
    assert expected_calibration_error(wrong, np.array([1, 1, 1, 1])) == pytest.approx(0.99)
    perfect = np.array([[1.0, 0.0]] * 4)
    assert expected_calibration_error(perfect, y) == pytest.approx(0.0)


def test_threshold_sweep_coverage_and_error():
    probs = np.array([[0.95, 0.05], [0.6, 0.4], [0.9, 0.1], [0.55, 0.45]])
    y = np.array([0, 1, 0, 0])
    rows = {r["threshold"]: r for r in threshold_sweep(probs, y, thresholds=(0.0, 0.8, 0.99))}
    assert rows[0.0]["coverage"] == 1.0 and rows[0.0]["error"] == pytest.approx(0.25)
    assert rows[0.8]["coverage"] == 0.5 and rows[0.8]["error"] == 0.0
    assert rows[0.99]["coverage"] == 0.0 and rows[0.99]["error"] is None


def test_build_report_has_required_fields():
    logits = np.array([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0]])
    r = build_report(logits, np.array([0, 1, 1]), ["a", "b"], temperature=2.0)
    for key in ("accuracy", "macro_f1", "precision", "recall", "confusion", "ece", "threshold_sweep", "temperature"):
        assert key in r
    json.dumps(r)


# ---- temperature fitting ----
def test_fit_temperature_cools_overconfident_and_sharpens_underconfident():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=400)
    noisy = np.eye(4)[y] * 1.0 + rng.normal(size=(400, 4))  # moderately informative logits
    assert fit_temperature(noisy * 8, y) > 3.0  # overconfident -> T > 1
    assert fit_temperature(noisy / 8, y) < 0.5  # underconfident -> T < 1
    t = fit_temperature(noisy, y)
    assert 0.05 <= t <= 20.0
    assert softmax(noisy, t).shape == (400, 4)


# ---- Router.load reads router_meta.json ----
@pytest.fixture
def meta_checkpoint(tmp_path):
    """Base-model files symlinked into a tmp dir so only router_meta.json differs."""
    if not MODEL_DIR.is_dir():
        pytest.skip("run router/download_base.py first")
    os.environ["HF_HUB_OFFLINE"] = "1"
    for f in MODEL_DIR.iterdir():
        (tmp_path / f.name).symlink_to(f)

    def write(**meta):
        (tmp_path / META_FILENAME).write_text(json.dumps(meta))
        return tmp_path

    return write


def test_router_load_uses_meta_temperature_and_trained_flag(meta_checkpoint):
    labels = load_labels(LABELS_PATH)
    ckpt = meta_checkpoint(trained=True, labels=labels, temperature=1.7, max_length=64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # trained checkpoint must not emit the UNTRAINED warning
        router = Router.load(ckpt, LABELS_PATH, device="cpu")
    assert router.temperature == 1.7 and router.max_length == 64 and not router.untrained
    explicit = Router.load(ckpt, LABELS_PATH, device="cpu", temperature=0.5)
    assert explicit.temperature == 0.5


def test_router_load_rejects_label_order_mismatch(meta_checkpoint):
    labels = load_labels(LABELS_PATH)
    ckpt = meta_checkpoint(trained=True, labels=list(reversed(labels)), temperature=1.0)
    with pytest.raises(ValueError, match="label order"):
        Router.load(ckpt, LABELS_PATH, device="cpu")
