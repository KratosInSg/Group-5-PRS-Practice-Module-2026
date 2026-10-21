"""Runtime wrapper for the frozen WS2 ConvNeXt + embedding-SVM hybrid.

Trusted local artifact only: joblib uses pickle and must never load untrusted files.
Input probabilities must be WS1's already calibrated ten-class ConvNeXt probabilities;
embedding must be the same 256-D PCA-normalised vector from VisionModel.embed().
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLASSES = (
    "paper", "plastic", "glass", "metal", "beverage_carton",
    "styrofoam", "e_waste", "textile", "food_waste", "other_general",
)


def temperature_scale(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1.0)) / t
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class HybridPredictor:
    def __init__(self, folder: Path):
        import joblib

        self.pipeline = joblib.load(folder / "svm_pipeline.joblib")
        self.config = json.loads((folder / "ensemble_config.json").read_text(encoding="utf-8"))
        if tuple(self.config["classes"]) != CLASSES:
            raise ValueError("Hybrid class order mismatch")
        if self.config["embedding_dim"] != 256 or self.config["cnn_weight"] != 0.5:
            raise ValueError("Unexpected hybrid configuration")
        self.svm_class_ids = np.asarray(self.pipeline.named_steps["svc"].classes_, dtype=int)
        if not np.array_equal(self.svm_class_ids, np.arange(10)):
            raise ValueError("SVM class order mismatch")

    def predict(self, cnn_probabilities: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        cnn = np.asarray(cnn_probabilities, dtype=np.float64).reshape(1, 10)
        emb = np.asarray(embedding, dtype=np.float64).reshape(1, 256)
        if not np.isfinite(cnn).all() or not np.isfinite(emb).all():
            raise ValueError("Non-finite inputs")
        if np.any(cnn < 0) or not np.allclose(cnn.sum(axis=1), 1, atol=2e-4):
            raise ValueError("CNN probabilities must sum to one")
        svm = self.pipeline.predict_proba(emb)
        svm = temperature_scale(svm, float(self.config["svm_temperature"]))
        return (0.5 * cnn + 0.5 * svm)[0]


def predict_photo(image, vision_model, hybrid: HybridPredictor) -> np.ndarray:
    """Return 10 probabilities in CLASSES order using the current WS1 ONNX model."""
    from app.vision import softmax

    x, _ = vision_model.preprocess(image.convert("RGB"))
    logits, pooled, _ = vision_model.session.run(None, {"image": x})
    cnn = softmax(np.asarray(logits[0], dtype=np.float32) / vision_model.temperature)
    embedding = vision_model.embed(np.asarray(pooled[0], dtype=np.float32))
    return hybrid.predict(cnn, np.asarray(embedding))
