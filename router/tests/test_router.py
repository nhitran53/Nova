"""Router smoke tests. Offline once the base model is downloaded (see download_base.py).

Run from /home/isaach/Nova/router: uv run pytest
"""
import math
import os
from pathlib import Path

import pytest

from router import Router, load_labels
from router.core import base_model_dir

HERE = Path(__file__).parent.parent
MODEL_DIR = Path(os.environ.get("ROUTER_MODEL_DIR", base_model_dir()))
LABELS = HERE / "labels.json"
TEXTS = ["check my oxygen level", "where is the rover", "what is the weather like on mars?", "tell me a joke"]

pytestmark = pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="run router/download_base.py first")


@pytest.fixture(scope="module")
def router():
    os.environ["HF_HUB_OFFLINE"] = "1"
    with pytest.warns(UserWarning, match="UNTRAINED"):
        return Router.load(MODEL_DIR, LABELS, device="cpu")


def test_num_labels_matches_labels_file(router):
    labels = load_labels(LABELS)
    assert router.model.config.num_labels == len(labels) == len(router.labels)
    assert router.model(**router.tokenizer("hi", return_tensors="pt")).logits.shape == (1, len(labels))


def test_untrained_flag_is_loud(router):
    r = router.predict("check my oxygen level")
    assert r["untrained"] is True and "UNTRAINED" in r["warning"]


def test_probs_sum_to_one_and_topk_sorted(router):
    full = router.probs(TEXTS)
    assert all(math.isclose(row.sum().item(), 1.0, abs_tol=1e-5) for row in full)
    for r in router.predict_batch(TEXTS, topk=5):
        ps = [t["p"] for t in r["topk"]]
        assert len(ps) == 5 and ps == sorted(ps, reverse=True)
        assert r["label"] == r["topk"][0]["label"] and r["confidence"] == ps[0]
        assert {t["label"] for t in r["topk"]} <= set(router.labels)


def test_latency_recorded(router):
    assert router.predict("hello")["latency_ms"] > 0


def test_batch_equals_single(router):
    batch = router.predict_batch(TEXTS, topk=len(router.labels))
    for text, b in zip(TEXTS, batch):
        s = router.predict(text, topk=len(router.labels))
        assert s["label"] == b["label"]
        sp = {t["label"]: t["p"] for t in s["topk"]}
        bp = {t["label"]: t["p"] for t in b["topk"]}
        assert all(math.isclose(sp[k], bp[k], abs_tol=1e-4) for k in sp)


def test_temperature_flattens_distribution(router):
    hot = router.predict("check my oxygen level", temperature=10.0)["confidence"]
    cold = router.predict("check my oxygen level", temperature=0.1)["confidence"]
    assert cold >= hot
    with pytest.raises(ValueError):
        router.predict("x", temperature=0)
