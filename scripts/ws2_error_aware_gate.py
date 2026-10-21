"""WS2-22: audit SVM-induced failures and evaluate validation-selected gating.

No test-label-driven selection. SVM uses ConvNeXt embeddings. All costs here are
material-only proxies, not WS3 final policy verdicts.
Run: uv run --with scikit-learn --with scipy --with joblib python scripts/ws2_error_aware_gate.py
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
BLUE = np.arange(5)


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    if len({r["path"] for r in records}) != len(records):
        raise ValueError(f"Duplicate paths in {path}")
    return {r["path"]: r for r in records}


def probability(records, paths):
    p = np.array([[float(records[path][f"p_{c}"]) for c in CLASSES] for path in paths])
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=2e-4):
        raise ValueError("Invalid probabilities")
    return p


def metrics(y, p, gate=None, threshold=None):
    pred = p.argmax(axis=1)
    blue_truth = np.isin(y, BLUE)
    pred_blue = np.isin(pred, BLUE)
    approved = pred_blue if threshold is None else p[:, BLUE].sum(axis=1) >= threshold
    return {
        "n": len(y),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(10), average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "costly_errors": int((approved & ~blue_truth).sum()),
        "nonblue_n": int((~blue_truth).sum()),
        "blue_recall": float((approved & blue_truth).sum() / blue_truth.sum()),
        "deferred_fraction": float((~approved).mean()) if threshold is not None else 0.0,
    }


def gated(swin, svm, margin):
    # SVM can contribute only when both models agree on the broad material
    # eligibility group and Swin is not confident in its top material.
    same_group = np.isin(swin.argmax(axis=1), BLUE) == np.isin(svm.argmax(axis=1), BLUE)
    use = same_group & (swin.max(axis=1) < margin)
    mixed = 0.5 * (swin + svm)
    return np.where(use[:, None], mixed, swin), use


def audit(paths, y, swin, svm, chosen, use):
    original = swin.argmax(axis=1)
    equal = (0.5 * (swin + svm)).argmax(axis=1)
    gated_pred = chosen.argmax(axis=1)
    output = []
    for i, path in enumerate(paths):
        bad = y[i] not in BLUE
        output.append({
            "path": path,
            "true": CLASSES[int(y[i])],
            "swin": CLASSES[int(original[i])],
            "svm": CLASSES[int(svm[i].argmax())],
            "equal": CLASSES[int(equal[i])],
            "gated": CLASSES[int(gated_pred[i])],
            "svm_used": int(use[i]),
            "swin_correct": int(original[i] == y[i]),
            "equal_correct": int(equal[i] == y[i]),
            "gated_correct": int(gated_pred[i] == y[i]),
            "svm_hurts": int(original[i] == y[i] and equal[i] != y[i]),
            "svm_rescues": int(original[i] != y[i] and equal[i] == y[i]),
            "new_costly": int(bad and original[i] not in BLUE and equal[i] in BLUE),
            "prevented_costly": int(bad and original[i] in BLUE and equal[i] not in BLUE),
            "swin_confidence": float(swin[i].max()),
            "svm_confidence": float(svm[i].max()),
            "swin_pblue": float(swin[i, BLUE].sum()),
            "svm_pblue": float(svm[i, BLUE].sum()),
        })
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=ROOT / "v0.2-local-data/predictions.csv")
    ap.add_argument("--embeddings", type=Path, default=ROOT / "v0.2-local-data/embeddings.npy")
    ap.add_argument("--manifest", type=Path, default=ROOT / "validation_partitions_v1.csv")
    ap.add_argument("--swin-val", type=Path, default=ROOT / "run3/swin_validation.csv")
    ap.add_argument("--swin-local-test", type=Path, default=ROOT / "run3/swin_local_test.csv")
    ap.add_argument("--artifact-dir", type=Path, default=Path("results/ws2/ensemble_handoff"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/ws2/ensemble_handoff/error_aware"))
    args = ap.parse_args()
    with args.predictions.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    index = {r["path"]: i for i, r in enumerate(rows)}
    if len(index) != len(rows):
        raise ValueError("Duplicate original paths")
    embeddings = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
    if embeddings.shape != (len(rows), 256):
        raise ValueError("Embedding alignment mismatch")
    manifest, swin_val, swin_test = read(args.manifest), read(args.swin_val), read(args.swin_local_test)
    if len(manifest) != 1556 or set(manifest) != set(swin_val):
        raise ValueError("Validation mismatch")
    test_paths = sorted(r["path"] for r in rows if r["source"] == "local" and r["split"] == "test")
    if len(test_paths) != 395 or set(test_paths) != set(swin_test):
        raise ValueError("Expected exactly 395 matching local test predictions")
    for path, rec in list(swin_val.items()) + list(swin_test.items()):
        original = rows[index[path]]
        if any(original[k] != rec[k] for k in ("source", "split", "group", "material")):
            raise ValueError(f"Metadata mismatch: {path}")
    for path, rec in manifest.items():
        original = rows[index[path]]
        if original["split"] != "val" or any(original[k] != rec[k] for k in ("source", "group", "material")):
            raise ValueError(f"Manifest mismatch: {path}")
    predictor = HybridPredictor(args.artifact_dir)
    val_paths = sorted(manifest)
    cal_paths = [p for p in val_paths if manifest[p]["partition"] == "calibration"]
    meta_paths = [p for p in val_paths if manifest[p]["partition"] == "meta_train" and manifest[p]["source"] == "local"]
    diag_paths = [p for p in val_paths if manifest[p]["partition"] == "diagnostic" and manifest[p]["source"] == "local"]
    if (len(cal_paths), len(meta_paths), len(diag_paths)) != (621, 76, 72):
        raise ValueError("Canonical partition counts changed")
    t = fit_temperature(
        probability(swin_val, cal_paths),
        np.array([CLASSES.index(manifest[p]["material"]) for p in cal_paths]),
    )
    def data(paths, records):
        y = np.array([CLASSES.index(rows[index[p]]["material"]) for p in paths])
        swin = temperature_scale(probability(records, paths), t)
        svm = temperature_scale(
            predictor.pipeline.predict_proba(embeddings[[index[p] for p in paths]]),
            float(predictor.config["svm_temperature"]),
        )
        return y, swin, svm

    ym, sm, vm = data(meta_paths, swin_val)
    baseline = metrics(ym, sm)
    # Predefined confidence margins; 0 means never use SVM.
    trials = []
    for margin in (0.0, 0.6, 0.7, 0.8, 0.9, 1.01):
        mixed, use = gated(sm, vm, margin)
        m = metrics(ym, mixed)
        trials.append({"margin": margin, "svm_used": int(use.sum()), **m})
    feasible = [t for t in trials if t["costly_errors"] <= baseline["costly_errors"]]
    chosen = max(feasible, key=lambda r: (r["macro_f1"], r["accuracy"], -r["costly_errors"], -r["svm_used"]))
    output = {
        "selected_margin": chosen["margin"],
        "meta_baseline": baseline,
        "meta_trials": trials,
        "swin_temperature": t,
        "evaluations": {},
        "limitations": [
            "This is a material-only safety proxy, not the policy verdict.",
            "Local test data were inspected previously; this is retrospective evaluation.",
            "Validation selection uses only 76 local meta-training images.",
            "Agreement gating is a heuristic and may not generalise.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, paths, records in (
        ("local_diagnostic", diag_paths, swin_val),
        ("local_test", test_paths, swin_test),
    ):
        y, swin, svm = data(paths, records)
        fused, used = gated(swin, svm, chosen["margin"])
        equal = 0.5 * (swin + svm)
        records_out = audit(paths, y, swin, svm, fused, used)
        with (args.output_dir / f"{split}_audit.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records_out[0]))
            writer.writeheader()
            writer.writerows(records_out)
        output["evaluations"][split] = {
            "swin": metrics(y, swin),
            "equal_vote": metrics(y, equal),
            "gated": metrics(y, fused),
            "svm_used": int(used.sum()),
            "svm_hurts": sum(r["svm_hurts"] for r in records_out),
            "svm_rescues": sum(r["svm_rescues"] for r in records_out),
            "new_costly": sum(r["new_costly"] for r in records_out),
            "prevented_costly": sum(r["prevented_costly"] for r in records_out),
            "fixed_blue_gate": {
                str(threshold): {
                    "swin": metrics(y, swin, threshold=threshold),
                    "equal_vote": metrics(y, equal, threshold=threshold),
                    "gated": metrics(y, fused, threshold=threshold),
                }
                for threshold in (0.8, 0.85, 0.9)
            },
        }
        print(split, json.dumps(output["evaluations"][split], indent=2))
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Validation-selected margin:", chosen["margin"])
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
