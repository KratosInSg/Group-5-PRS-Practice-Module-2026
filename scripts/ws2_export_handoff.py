"""Fit, export, reload and evaluate WS2 hybrid on all 395 Singapore test photos.

uv run --with scikit-learn --with scipy --with joblib python scripts/ws2_export_handoff.py

Requires the original aligned CSV/NPY files and canonical validation manifest.
Does not change original train/val/test partitions or fit on test labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ws2_handoff_inference import CLASSES, HybridPredictor, temperature_scale
from ws2_public_ensemble import fit_temperature

BLUE = set(CLASSES[:5])


def metrics(y: np.ndarray, probs: np.ndarray) -> dict[str, object]:
    pred = probs.argmax(axis=1)
    p, r, f, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(10), zero_division=0
    )
    nonblue = np.array([CLASSES[int(i)] not in BLUE for i in y])
    predicted_blue = np.array([CLASSES[int(i)] in BLUE for i in pred])
    return {
        "n": len(y),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(10), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "correct": int((pred == y).sum()),
        "costly_error_count": int((nonblue & predicted_blue).sum()),
        "nonblue_count": int(nonblue.sum()),
        "costly_error_rate": float((nonblue & predicted_blue).sum() / nonblue.sum()),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]),
                   "f1": float(f[i]), "support": int(support[i])}
            for i, name in enumerate(CLASSES)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("data/ws2/v0.2-local-data/predictions.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/ws2/v0.2-local-data/embeddings.npy"))
    parser.add_argument("--manifest", type=Path, default=Path("data/ws2/validation_partitions_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/ws2/ensemble_handoff"))
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len(rows) != 9785 or len(manifest_rows) != 1556:
        raise ValueError("Unexpected original data or validation manifest count")
    if len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image paths")
    embeddings = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
    if embeddings.shape != (len(rows), 256):
        raise ValueError("Embedding shape mismatch")
    index = {r["path"]: i for i, r in enumerate(rows)}
    if any(r["path"] not in index or rows[index[r["path"]]]["split"] != "val" for r in manifest_rows):
        raise ValueError("Manifest does not match validation rows")
    for row in manifest_rows:
        original = rows[index[row["path"]]]
        if any(original[k] != row[k] for k in ("source", "group", "material")):
            raise ValueError("Manifest metadata mismatch")
    train_idx = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    cal_idx = np.array([index[r["path"]] for r in manifest_rows if r["partition"] == "calibration"])
    test_idx = np.array([i for i, r in enumerate(rows) if r["split"] == "test" and r["source"] == "local"])
    if (len(train_idx), len(cal_idx), len(test_idx)) != (6557, 621, 395):
        raise ValueError("Unexpected split counts")
    train_groups = {(rows[int(i)]["source"], rows[int(i)]["group"]) for i in train_idx}
    cal_groups = {(rows[int(i)]["source"], rows[int(i)]["group"]) for i in cal_idx}
    test_groups = {(rows[int(i)]["source"], rows[int(i)]["group"]) for i in test_idx}
    if train_groups & cal_groups or train_groups & test_groups or cal_groups & test_groups:
        raise ValueError("Physical-item leakage")

    y_train = np.array([CLASSES.index(rows[int(i)]["material"]) for i in train_idx])
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=0.1, class_weight="balanced", probability=True, random_state=42),
    )
    svm.fit(embeddings[train_idx], y_train)
    cal_y = np.array([CLASSES.index(rows[int(i)]["material"]) for i in cal_idx])
    t = fit_temperature(svm.predict_proba(embeddings[cal_idx]), cal_y)

    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(svm, args.output / "svm_pipeline.joblib")
    cfg = {
        "classes": list(CLASSES), "embedding_dim": 256, "cnn_weight": 0.5,
        "svm_temperature": t, "svm": {"kernel": "linear", "C": 0.1,
        "class_weight": "balanced", "probability": True, "random_state": 42},
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "embedding_source": "WS1 VisionModel.embed: PCA projection followed by L2 normalisation",
        "cnn_probability_source": "WS1 ONNX logits / config temperature, softmax",
        "training_rows": len(train_idx), "temperature_calibration_rows": len(cal_idx),
        "note": "Material probabilities only. Condition/contamination and final verdict belong to WS3.",
    }
    (args.output / "ensemble_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    predictor = HybridPredictor(args.output)

    cnn_test = np.array([[float(rows[int(i)][f"p_{name}"]) for name in CLASSES] for i in test_idx])
    svm_test = temperature_scale(svm.predict_proba(embeddings[test_idx]), t)
    hybrid = 0.5 * cnn_test + 0.5 * svm_test
    loaded = np.stack([predictor.predict(cnn_test[j], embeddings[int(i)]) for j, i in enumerate(test_idx)])
    max_delta = float(np.max(np.abs(hybrid - loaded)))
    if not np.allclose(hybrid, loaded, atol=1e-10):
        raise AssertionError(f"Saved artifact predictions differ: {max_delta}")

    y = np.array([CLASSES.index(rows[int(i)]["material"]) for i in test_idx])
    baseline, ensemble = metrics(y, cnn_test), metrics(y, loaded)
    report = {
        "split": "original Singapore local test",
        "source": "local",
        "n": len(test_idx),
        "baseline": baseline,
        "hybrid": ensemble,
        "max_reload_probability_difference": max_delta,
        "svm_temperature": t,
        "model_file_bytes": (args.output / "svm_pipeline.joblib").stat().st_size,
        "limitations": [
            "Test set has been inspected in previous research; this is a retrospective frozen-configuration comparison.",
            "Costly-error rate is a material-only proxy, not the final contamination-aware disposal verdict.",
            "CNN probabilities come from WS1's saved prediction CSV; image-to-embedding parity must be independently verified against the deployed ONNX model.",
            "Only load the joblib file from a trusted source; pickle deserialisation can execute code.",
        ],
    }
    (args.output / "singapore_test_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "singapore_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "true_material", "cnn_predicted", "hybrid_predicted", "cnn_correct", "hybrid_correct", "cnn_costly", "hybrid_costly"])
        for j, i in enumerate(test_idx):
            true = CLASSES[int(y[j])]
            cnn_label = CLASSES[int(cnn_test[j].argmax())]
            hybrid_label = CLASSES[int(loaded[j].argmax())]
            writer.writerow([
                rows[int(i)]["path"], true, cnn_label, hybrid_label,
                int(cnn_label == true), int(hybrid_label == true),
                int(true not in BLUE and cnn_label in BLUE),
                int(true not in BLUE and hybrid_label in BLUE),
            ])
    print(json.dumps({
        "cnn": {k: baseline[k] for k in ("macro_f1", "accuracy", "costly_error_count", "nonblue_count", "costly_error_rate")},
        "hybrid": {k: ensemble[k] for k in ("macro_f1", "accuracy", "costly_error_count", "nonblue_count", "costly_error_rate")},
        "svm_temperature": t,
        "sklearn_version": sklearn.__version__,
        "svm_bytes": report["model_file_bytes"],
        "reload_max_difference": max_delta,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
