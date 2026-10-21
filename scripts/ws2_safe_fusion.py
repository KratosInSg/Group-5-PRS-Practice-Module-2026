"""WS2 safety-constrained, class-aware fusion experiment.

Fits a SMALL 2-parameter fusion (blue-material vs nonblue SVM weight) on
canonical meta_train LOCAL data, with predeclared safety constraints.
Calibration/SVM are frozen from WS2 handoff. Diagnostic local and public test
are evaluation-only. The local test is retrospective and NOT used for selection.

uv run --with scikit-learn --with joblib python scripts/ws2_safe_fusion.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from ws2_handoff_inference import CLASSES, HybridPredictor, temperature_scale

BLUE = np.arange(5)
GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
ROOT = Path("data/ws2")


def fusion(cnn: np.ndarray, svm: np.ndarray, blue_weight: float, other_weight: float) -> np.ndarray:
    weights = np.array([blue_weight] * 5 + [other_weight] * 5)
    combined = (1 - weights) * cnn + weights * svm
    return combined / combined.sum(axis=1, keepdims=True)


def report(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    pred = p.argmax(axis=1)
    nonblue = ~np.isin(y, BLUE)
    predicted_blue = np.isin(pred, BLUE)
    return {
        "n": len(y),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(10), average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "costly_errors": int((nonblue & predicted_blue).sum()),
        "nonblue_n": int(nonblue.sum()),
        "costly_rate": float((nonblue & predicted_blue).sum() / nonblue.sum()) if nonblue.any() else None,
        "blue_recall": float((np.isin(y, BLUE) & predicted_blue).sum() / np.isin(y, BLUE).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=ROOT / "v0.2-local-data/predictions.csv")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "v0.2-local-data/embeddings.npy")
    parser.add_argument("--manifest", type=Path, default=ROOT / "validation_partitions_v1.csv")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/ws2/ensemble_handoff"))
    parser.add_argument("--output", type=Path, default=Path("results/ws2/ensemble_handoff/safe_fusion.json"))
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    embeddings = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
    if embeddings.shape != (len(rows), 256):
        raise ValueError("Invalid embeddings shape")
    if len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate paths")
    indices = {r["path"]: i for i, r in enumerate(rows)}
    if len(manifest) != 1556 or any(r["path"] not in indices for r in manifest):
        raise ValueError("Invalid canonical manifest")
    for r in manifest:
        original = rows[indices[r["path"]]]
        if original["split"] != "val" or any(r[k] != original[k] for k in ("material", "source", "group")):
            raise ValueError("Manifest metadata mismatch")
    selected = {
        "meta_local": [indices[r["path"]] for r in manifest if r["partition"] == "meta_train" and r["source"] == "local"],
        "diagnostic_local": [indices[r["path"]] for r in manifest if r["partition"] == "diagnostic" and r["source"] == "local"],
        "test_local": [i for i, r in enumerate(rows) if r["split"] == "test" and r["source"] == "local"],
        "test_public": [i for i, r in enumerate(rows) if r["split"] == "test" and r["source"] != "local"],
    }
    if tuple(len(selected[k]) for k in ("meta_local", "diagnostic_local", "test_local", "test_public")) != (76, 72, 395, 1277):
        raise ValueError("Unexpected split counts")
    predictor = HybridPredictor(args.artifact_dir)
    y_all = np.array([CLASSES.index(r["material"]) for r in rows])
    p_all = np.array([[float(r[f"p_{c}"]) for c in CLASSES] for r in rows])
    eval_ids = np.array(sorted(set(sum(selected.values(), []))))
    svm_probs = temperature_scale(
        predictor.pipeline.predict_proba(embeddings[eval_ids]),
        float(predictor.config["svm_temperature"]),
    )
    svm_by_id = {int(i): svm_probs[j] for j, i in enumerate(eval_ids)}

    def subset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = selected[name]
        return y_all[ids], p_all[ids], np.stack([svm_by_id[i] for i in ids])

    y_meta, c_meta, s_meta = subset("meta_local")
    baseline_meta = report(y_meta, c_meta)
    trials = []
    for blue_weight in GRID:
        for other_weight in GRID:
            pred = fusion(c_meta, s_meta, blue_weight, other_weight)
            m = report(y_meta, pred)
            trials.append({"blue_weight": blue_weight, "other_weight": other_weight, **m})
    # Safety constraint: no more false-blue errors than CNN on meta-training;
    # maximise macro-F1, then accuracy, then minimise SVM contribution.
    feasible = [t for t in trials if t["costly_errors"] <= baseline_meta["costly_errors"]]
    if not feasible:
        raise AssertionError("CNN-only candidate should always be feasible")
    chosen = max(
        feasible,
        key=lambda t: (t["macro_f1"], t["accuracy"], -t["blue_weight"] - t["other_weight"]),
    )
    results = {
        "selection": {"baseline": baseline_meta, "chosen": chosen, "candidates": len(trials)},
        "evaluation": {},
        "limitations": [
            "Class-aware fusion has only two weights: blue-material and other-material SVM contribution.",
            "Weights were selected on 76 local meta-training images, with high variance.",
            "Singapore and public test data have been inspected previously; comparisons are retrospective.",
            "This evaluates material predictions, not contamination-aware end-to-end verdicts.",
            "The original SVM shares ConvNeXt features and cannot correct errors outside that representation.",
        ],
    }
    for name in ("diagnostic_local", "test_local", "test_public"):
        y, c, s = subset(name)
        results["evaluation"][name] = {
            "cnn": report(y, c),
            "selected_fusion": report(y, fusion(c, s, chosen["blue_weight"], chosen["other_weight"])),
            "equal_50_50": report(y, fusion(c, s, 0.5, 0.5)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("Selected validation-only weights:", chosen["blue_weight"], chosen["other_weight"])
    print("Selection baseline:", json.dumps(baseline_meta))
    for name, comparison in results["evaluation"].items():
        print(name)
        for model, m in comparison.items():
            print(f"  {model:18s} F1={m['macro_f1']:.4f} accuracy={m['accuracy']:.4f} costly={m['costly_errors']}/{m['nonblue_n']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
