"""WS2 handoff sensitivity: 50/50 vs 75/25 fusion and fixed blue-material gates.

This is retrospective exploration on the previously inspected 395 local test images.
Never select a deployment threshold using these test results.

uv run --with scikit-learn --with joblib python scripts/ws2_handoff_sensitivity.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from ws2_handoff_inference import CLASSES, HybridPredictor

BLUE = np.arange(5)


def evaluate(y: np.ndarray, probs: np.ndarray, threshold: float | None) -> dict:
    pred = probs.argmax(axis=1)
    true_blue = np.isin(y, BLUE)
    pred_blue = np.isin(pred, BLUE)
    if threshold is None:
        approve = pred_blue
        deferred = np.zeros(len(y), dtype=bool)
    else:
        # Material-only policy proxy: blue only if P(blue materials) >= threshold.
        approve = probs[:, BLUE].sum(axis=1) >= threshold
        deferred = ~approve
    bad = int((approve & ~true_blue).sum())
    eligible = int(true_blue.sum())
    return {
        "macro_f1_material": float(f1_score(y, pred, labels=np.arange(10), average="macro", zero_division=0)),
        "accuracy_material": float(np.mean(pred == y)),
        "costly_errors": bad,
        "nonblue_n": int((~true_blue).sum()),
        "costly_error_rate": bad / int((~true_blue).sum()),
        "blue_recall": float((approve & true_blue).sum() / eligible),
        "deferred_n": int(deferred.sum()),
        "deferred_fraction": float(deferred.mean()),
        "blue_precision": float((approve & true_blue).sum() / approve.sum()) if approve.any() else None,
        "note": "Ungated: top-class material proxy. Gated: sum of five blue-material probabilities; deferred is not general waste.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("data/ws2/v0.2-local-data/predictions.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/ws2/v0.2-local-data/embeddings.npy"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/ws2/ensemble_handoff"))
    parser.add_argument("--output", type=Path, default=Path("results/ws2/ensemble_handoff/sensitivity.json"))
    args = parser.parse_args()

    with args.predictions.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    emb = np.load(args.embeddings, allow_pickle=False, mmap_mode="r")
    if emb.shape != (len(rows), 256):
        raise ValueError("CSV/embedding alignment mismatch")
    ids = [i for i, r in enumerate(rows) if r["split"] == "test" and r["source"] == "local"]
    if len(ids) != 395:
        raise ValueError(f"Expected 395 local test photos, found {len(ids)}")
    y = np.array([CLASSES.index(rows[i]["material"]) for i in ids])
    cnn = np.array([[float(rows[i][f"p_{c}"]) for c in CLASSES] for i in ids])
    predictor = HybridPredictor(args.artifact_dir)
    # Derive SVM probabilities from the saved pipeline and frozen calibration.
    svm = predictor.pipeline.predict_proba(np.asarray(emb[ids]))
    from ws2_handoff_inference import temperature_scale
    svm = temperature_scale(svm, float(predictor.config["svm_temperature"]))
    variants = {
        "cnn_only": cnn,
        "hybrid_50_50": 0.5 * cnn + 0.5 * svm,
        "hybrid_75_25": 0.75 * cnn + 0.25 * svm,
    }
    result = {
        "n_test": 395,
        "purpose": "retrospective sensitivity analysis, not test-based model selection",
        "variants": {},
    }
    for name, p in variants.items():
        result["variants"][name] = {
            "ungated": evaluate(y, p, None),
            "gate_0_80": evaluate(y, p, 0.80),
            "gate_0_85": evaluate(y, p, 0.85),
            "gate_0_90": evaluate(y, p, 0.90),
            "gate_0_95": evaluate(y, p, 0.95),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Model               Gate  MacroF1 Accuracy  Costly  BlueRecall Deferred")
    for name, metrics in result["variants"].items():
        for gate, m in metrics.items():
            print(
                f"{name:19s} {gate:10s} {m['macro_f1_material']:.4f}  "
                f"{m['accuracy_material']:.4f}  {m['costly_errors']:2d}/{m['nonblue_n']}  "
                f"{m['blue_recall']:.3f}      {m['deferred_fraction']:.3f}"
            )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
