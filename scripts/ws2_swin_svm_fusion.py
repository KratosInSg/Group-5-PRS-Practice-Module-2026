"""Compare Swin-T + ConvNeXt-embedding SVM against ConvNeXt alone.

Uses frozen SVM artifact and temperature, Swin temperature from canonical
calibration, and local meta_train for weight selection. Diagnostic local and
public test are evaluation-only. Requires Swin test predictions for local
395 images to evaluate the full Singapore test; otherwise skips that split.

uv run --with scikit-learn --with scipy --with joblib python scripts/ws2_swin_svm_fusion.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from ws2_handoff_inference import CLASSES, HybridPredictor, temperature_scale
from ws2_public_ensemble import fit_temperature

ROOT = Path("data/ws2")
BLUE = set(CLASSES[:5])
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len({r["path"] for r in rows}):
        raise ValueError(f"Duplicate paths: {path}")
    return {r["path"]: r for r in rows}


def probs(rows: list[dict[str, str]], prefix: str) -> np.ndarray:
    p = np.array([[float(r[prefix + c]) for c in CLASSES] for r in rows])
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=2e-4):
        raise ValueError("Invalid probability matrix")
    return p


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    pred = p.argmax(axis=1)
    nonblue = np.array([CLASSES[int(i)] not in BLUE for i in y])
    blue = np.array([CLASSES[int(i)] in BLUE for i in pred])
    return {
        "n": len(y),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(10), average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "costly_errors": int((nonblue & blue).sum()),
        "nonblue_n": int(nonblue.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=ROOT / "v0.2-local-data/predictions.csv")
    ap.add_argument("--embeddings", type=Path, default=ROOT / "v0.2-local-data/embeddings.npy")
    ap.add_argument("--manifest", type=Path, default=ROOT / "validation_partitions_v1.csv")
    ap.add_argument("--swin-val", type=Path, default=ROOT / "run3/swin_validation.csv")
    ap.add_argument("--swin-public", type=Path, default=ROOT / "run3/pred_E2_public_plus_local_swin_t_public_test.csv")
    ap.add_argument("--swin-local-test", type=Path, default=None)
    ap.add_argument("--artifact-dir", type=Path, default=Path("results/ws2/ensemble_handoff"))
    ap.add_argument("--output", type=Path, default=Path("results/ws2/ensemble_handoff/swin_svm_fusion.json"))
    args = ap.parse_args()

    with args.predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate CNN paths")
    index = {r["path"]: i for i, r in enumerate(rows)}
    embeddings = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
    if embeddings.shape != (len(rows), 256):
        raise ValueError("Embedding alignment mismatch")
    manifest, swin_val, swin_public = read(args.manifest), read(args.swin_val), read(args.swin_public)
    if len(manifest) != 1556 or set(manifest) != set(swin_val):
        raise ValueError("Canonical validation mismatch")
    val_paths = sorted(manifest)
    public_paths = sorted(r["path"] for r in rows if r["split"] == "test" and r["source"] != "local")
    if set(public_paths) != set(swin_public):
        raise ValueError("Public Swin paths mismatch")
    for path in val_paths:
        if path not in index or any(manifest[path][k] != other[path][k] for other in (rows_as_dict(rows), swin_val) for k in ("source", "group", "material")):
            raise ValueError(f"Validation metadata mismatch: {path}")
    for path in public_paths:
        if rows[index[path]]["material"] != swin_public[path]["true"]:
            raise ValueError(f"Public label mismatch: {path}")

    predictor = HybridPredictor(args.artifact_dir)
    def get_svm(paths: list[str]) -> np.ndarray:
        return temperature_scale(
            predictor.pipeline.predict_proba(embeddings[[index[p] for p in paths]]),
            float(predictor.config["svm_temperature"]),
        )
    yv = np.array([CLASSES.index(manifest[p]["material"]) for p in val_paths])
    partition = np.array([manifest[p]["partition"] for p in val_paths])
    local = np.array([manifest[p]["source"] == "local" for p in val_paths])
    cal = partition == "calibration"
    meta = (partition == "meta_train") & local
    diag = (partition == "diagnostic") & local
    raw_swin_val = probs([swin_val[p] for p in val_paths], "p_")
    temp = fit_temperature(raw_swin_val[cal], yv[cal])
    swin_val_p = temperature_scale(raw_swin_val, temp)
    svm_val_p = get_svm(val_paths)
    cnn_val_p = probs([rows[index[p]] for p in val_paths], "p_")
    trials = []
    for w in WEIGHTS:
        mixed = w * swin_val_p + (1 - w) * svm_val_p
        trials.append({"swin_weight": w, **metrics(yv[meta], mixed[meta])})
    selected = max(trials, key=lambda t: (t["macro_f1"], -t["costly_errors"]))["swin_weight"]

    evaluations = {}
    def evaluate(name: str, paths: list[str], swin_p: np.ndarray, y: np.ndarray) -> None:
        svm_p = get_svm(paths)
        cnn_p = probs([rows[index[p]] for p in paths], "p_")
        variants = {
            "convnext": cnn_p,
            "swin": swin_p,
            "embedding_svm": svm_p,
            "swin_svm_equal": 0.5 * (swin_p + svm_p),
            "swin_svm_selected": selected * swin_p + (1 - selected) * svm_p,
        }
        evaluations[name] = {label: metrics(y, p) for label, p in variants.items()}
        print(name, len(paths))
        for label, result in evaluations[name].items():
            print(f"  {label:20s} F1={result['macro_f1']:.4f} accuracy={result['accuracy']:.4f} costly={result['costly_errors']}/{result['nonblue_n']}")

    diag_paths = [p for p, mask in zip(val_paths, diag) if mask]
    evaluate("local_diagnostic", diag_paths, swin_val_p[diag], yv[diag])
    public_raw = probs([swin_public[p] for p in public_paths], "")
    public_swin_p = temperature_scale(public_raw, temp)
    public_y = np.array([CLASSES.index(rows[index[p]]["material"]) for p in public_paths])
    evaluate("public_test", public_paths, public_swin_p, public_y)

    if args.swin_local_test is not None:
        local_swin = read(args.swin_local_test)
        local_paths = sorted(r["path"] for r in rows if r["split"] == "test" and r["source"] == "local")
        if set(local_swin) != set(local_paths):
            raise ValueError("Local Swin paths do not match 395 local test photos")
        for p in local_paths:
            if rows[index[p]]["material"] != local_swin[p].get("true", local_swin[p].get("material")):
                raise ValueError(f"Local label mismatch: {p}")
        raw = probs([local_swin[p] for p in local_paths], "" if "paper" in local_swin[local_paths[0]] else "p_")
        y = np.array([CLASSES.index(rows[index[p]]["material"]) for p in local_paths])
        evaluate("local_test", local_paths, temperature_scale(raw, temp), y)
    else:
        print("Local 395-image Swin test predictions not supplied; no local-test ensemble claim.")

    result = {
        "swin_temperature": temp,
        "selected_swin_weight": selected,
        "meta_trials": trials,
        "evaluation": evaluations,
        "limitations": [
            "SVM is trained on ConvNeXt embeddings, not Swin embeddings.",
            "Both ConvNeXt and Swin must run for live inference.",
            "Selection is on local meta_train only; public and local test are retrospective.",
            "No local test comparison is available without Swin predictions for the same 395 images.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {args.output}")


def rows_as_dict(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {r["path"]: r for r in rows}


if __name__ == "__main__":
    main()
