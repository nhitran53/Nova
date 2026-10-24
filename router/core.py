"""ModernBERT sequence-classifier router (text in -> label probabilities out).

The classification head is UNTRAINED until a training script writes
`router_meta.json` ({"trained": true, ...}) next to the saved model. Without
that marker every prediction carries `untrained: true` and a warning: labels
from a random head are meaningless.
"""
import json
import os
import time
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

META_FILENAME = "router_meta.json"
DEFAULT_MODELS_DIR = "/home/isaach/models/Nova/router"  # trained checkpoints live outside the repo
UNTRAINED_WARNING = (
    "UNTRAINED: classification head is randomly initialised "
    "(no router_meta.json with trained=true); labels/confidences are meaningless."
)


def models_dir():
    """Directory holding trained checkpoints (v1, ...): $NOVA_MODELS_DIR, else DEFAULT_MODELS_DIR."""
    return Path(os.environ.get("NOVA_MODELS_DIR", DEFAULT_MODELS_DIR))


def base_model_dir():
    """Pretrained ModernBERT-base (downloaded by download_base.py): <models_dir>/ModernBERT-base."""
    return models_dir() / "ModernBERT-base"


def load_labels(labels_path):
    """Read labels.json -> list[str]. Accepts {"labels": [...]} or a bare list."""
    data = json.loads(Path(labels_path).read_text())
    labels = data["labels"] if isinstance(data, dict) else data
    if not labels or len(set(labels)) != len(labels) or not all(isinstance(x, str) for x in labels):
        raise ValueError(f"{labels_path}: labels must be a non-empty list of unique strings")
    return list(labels)


def resolve_device(device="auto"):
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def read_meta(model_dir):
    """router_meta.json contents, or {} if absent/not a local dir."""
    path = Path(model_dir) / META_FILENAME
    return json.loads(path.read_text()) if path.is_file() else {}


class Router:
    def __init__(self, model, tokenizer, labels, device, meta, temperature=1.0, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        self.device = device
        self.meta = meta
        self.temperature = temperature
        self.max_length = max_length
        self.untrained = not meta.get("trained", False)

    @classmethod
    def load(cls, model_dir_or_id, labels_path, device="auto", temperature=None, max_length=None, seed=0):
        """Load a ModernBERT checkpoint (local dir or HF id) with a len(labels)-way head.

        `temperature` / `max_length` default to the values in router_meta.json (written by train.py),
        else 1.0 / 128. A meta file whose label order differs from labels.json is rejected.
        `seed` fixes the random head init so untrained outputs are at least reproducible.
        """
        labels = load_labels(labels_path)
        dev = resolve_device(device)
        meta = read_meta(model_dir_or_id)
        if "labels" in meta and meta["labels"] != labels:
            raise ValueError(f"{model_dir_or_id}: trained label order {meta['labels']} != {labels_path} {labels}")
        if temperature is None:
            temperature = meta.get("temperature", 1.0)
        if max_length is None:
            max_length = meta.get("max_length", 128)
        tokenizer = AutoTokenizer.from_pretrained(model_dir_or_id)
        torch.manual_seed(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # HF "newly initialized head" noise; we warn ourselves
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir_or_id,
                num_labels=len(labels),
                id2label=dict(enumerate(labels)),
                label2id={name: i for i, name in enumerate(labels)},
            )
        model.eval().to(dev)
        router = cls(model, tokenizer, labels, dev, meta, temperature, max_length)
        if router.untrained:
            warnings.warn(UNTRAINED_WARNING, stacklevel=2)
        return router

    @torch.inference_mode()
    def predict_batch(self, texts, topk=3, temperature=None):
        """Predict for a list of texts. Per-item `latency_ms` is the batch wall time / len(texts)."""
        temp = self.temperature if temperature is None else temperature
        if temp <= 0:
            raise ValueError("temperature must be > 0")
        k = min(topk, len(self.labels))
        start = time.perf_counter()
        enc = self.tokenizer(
            list(texts), return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        ).to(self.device)
        logits = self.model(**enc).logits.float()
        probs = torch.softmax(logits / temp, dim=-1)
        top_p, top_i = probs.topk(k, dim=-1)
        top_p, top_i = top_p.cpu().tolist(), top_i.cpu().tolist()
        _sync(self.device)
        per_item_ms = (time.perf_counter() - start) * 1000 / len(texts)
        results = []
        for ps, idxs in zip(top_p, top_i):
            ranked = [{"label": self.labels[i], "p": p} for i, p in zip(idxs, ps)]
            result = {
                "label": ranked[0]["label"],
                "confidence": ranked[0]["p"],
                "topk": ranked,
                "latency_ms": per_item_ms,
                "untrained": self.untrained,
            }
            if self.untrained:
                result["warning"] = UNTRAINED_WARNING
            results.append(result)
        return results

    def predict(self, text, topk=3, temperature=None):
        return self.predict_batch([text], topk=topk, temperature=temperature)[0]

    def probs(self, texts, temperature=None):
        """Full softmax distribution, shape [len(texts), num_labels] (used by tests/calibration)."""
        temp = self.temperature if temperature is None else temperature
        with torch.inference_mode():
            enc = self.tokenizer(
                list(texts), return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            ).to(self.device)
            return torch.softmax(self.model(**enc).logits.float() / temp, dim=-1).cpu()
