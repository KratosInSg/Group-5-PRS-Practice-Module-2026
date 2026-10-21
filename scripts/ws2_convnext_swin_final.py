"""WS2: ConvNeXt-T + Swin-T fusion, canonical validation selection, local test audit.

No test labels used to select fusion weight. Safety rule is a fixed ablation.
Run:
uv run --with scikit-learn --with scipy python scripts/ws2_convnext_swin_final.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from ws2_handoff_inference import CLASSES, temperature_scale
from ws2_public_ensemble import fit_temperature

BLUE = np.arange(5)
ROOT = Path("data/ws2")


def load(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len({r["path"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate paths: {path}")
    return {r["path"]: r for r in rows}


def matrix(rows, paths):
    p = np.asarray([[float(rows[path][f"p_{c}"]) for c in CLASSES] for path in paths])
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=2e-4):
        raise ValueError("Invalid probability matrix")
    return p


def scores(y, p):
    pred = p.argmax(axis=1)
    truth_blue = np.isin(y, BLUE)
    pred_blue = np.isin(pred, BLUE)
    return {
        "n": len(y),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(10), average="macro", zero_division=0)),
        "accuracy": float(np.mean(pred == y)),
        "costly_errors": int((~truth_blue & pred_blue).sum()),
        "nonblue_n": int((~truth_blue).sum()),
    }


def conservative(c, s):
    # Predeclared policy: when the CNNs disagree about material eligibility,
    # use ConvNeXt rather than allowing Swin to flip the broad bin category.
    same_group = np.isin(c.argmax(1), BLUE) == np.isin(s.argmax(1), BLUE)
    return np.where(same_group[:, None], 0.5 * (c + s), c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", type=Path, default=ROOT / "v0.2-local-data/predictions.csv")
    ap.add_argument("--manifest", type=Path, default=ROOT / "validation_partitions_v1.csv")
    ap.add_argument("--swin-val", type=Path, default=ROOT / "run3/swin_validation.csv")
    ap.add_argument("--swin-local-test", type=Path, default=ROOT / "run3/swin_local_test.csv")
    ap.add_argument("--output", type=Path, default=Path("results/ws2/ensemble_handoff/convnext_swin_final.json"))
    args = ap.parse_args()
    cnn, manifest, sv, st = (load(p) for p in (args.cnn, args.manifest, args.swin_val, args.swin_local_test))
    if len(manifest) != 1556 or set(manifest) != set(sv):
        raise ValueError("Validation alignment mismatch")
    test_paths = sorted(p for p, r in cnn.items() if r["split"] == "test" and r["source"] == "local")
    if len(test_paths) != 395 or set(test_paths) != set(st):
        raise ValueError("Expected exactly 395 local Swin test rows")
    for source in (sv, st):
        for p, r in source.items():
            if p not in cnn or any(cnn[p][key] != r[key] for key in ("source", "split", "material", "group")):
                raise ValueError(f"Metadata mismatch: {p}")
    for p, r in manifest.items():
        if p not in cnn or any(cnn[p][key] != r[key] for key in ("source", "group", "material")):
            raise ValueError(f"Manifest mismatch: {p}")

    val_paths = sorted(manifest)
    yv = np.array([CLASSES.index(manifest[p]["material"]) for p in val_paths])
    cal = np.array([manifest[p]["partition"] == "calibration" for p in val_paths])
    meta = np.array([manifest[p]["partition"] == "meta_train" and manifest[p]["source"] == "local" for p in val_paths])
    diag = np.array([manifest[p]["partition"] == "diagnostic" and manifest[p]["source"] == "local" for p in val_paths])
    if (int(cal.sum()), int(meta.sum()), int(diag.sum())) != (621, 76, 72):
        raise ValueError("Canonical split changed")
    raw = matrix(sv, val_paths)
    t = fit_temperature(raw[cal], yv[cal])
    cs, ss = matrix(cnn, val_paths), temperature_scale(raw, t)
    candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
    trials = []
    for w in candidates:
        p = w * cs + (1 - w) * ss
        trials.append({"convnext_weight": w, **scores(yv[meta], p[meta])})
    chosen = max(trials, key=lambda r: (r["macro_f1"], -r["costly_errors"], r["accuracy"]))["convnext_weight"]

    def evaluate(paths, sw, name):
        y = np.array([CLASSES.index(cnn[p]["material"]) for p in paths])
        c = matrix(cnn, paths)
        s = temperature_scale(matrix(sw, paths), t)
        methods = {
            "convnext": c,
            "swin": s,
            "equal_vote": 0.5 * (c + s),
            "selected_vote": chosen * c + (1 - chosen) * s,
            "conservative_vote": conservative(c, s),
        }
        result = {key: scores(y, p) for key, p in methods.items()}
        print(name)
        for key, v in result.items():
            print(f"  {key:20s} F1={v['macro_f1']:.4f} accuracy={v['accuracy']:.4f} costly={v['costly_errors']}/{v['nonblue_n']}")
        return result, y, methods

    diagnostic_paths = [p for p, keep in zip(val_paths, diag) if keep]
    diagnostic, _, _ = evaluate(diagnostic_paths, sv, "local_diagnostic")
    test, y, methods = evaluate(test_paths, st, "local_test")
    pred_c = methods["convnext"].argmax(1)
    pred_s = methods["swin"].argmax(1)
    pred_e = methods["equal_vote"].argmax(1)
    audit_path = args.output.with_name("convnext_swin_test_audit.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "true", "convnext", "swin", "equal_vote", "convnext_correct", "swin_correct", "equal_correct", "convnext_costly", "swin_costly", "equal_costly"))
        for i, path in enumerate(test_paths):
            nonblue = y[i] not in BLUE
            writer.writerow((
                path, CLASSES[int(y[i])], CLASSES[int(pred_c[i])], CLASSES[int(pred_s[i])],
                CLASSES[int(pred_e[i])], int(pred_c[i] == y[i]), int(pred_s[i] == y[i]),
                int(pred_e[i] == y[i]), int(nonblue and pred_c[i] in BLUE),
                int(nonblue and pred_s[i] in BLUE), int(nonblue and pred_e[i] in BLUE),
            ))
    report = {
        "swin_temperature": t, "selected_convnext_weight": chosen,
        "meta_trials": trials, "local_diagnostic": diagnostic, "local_test": test,
        "audit": str(audit_path),
        "limitations": [
            "Previously inspected Singapore test set: retrospective comparison, not fresh validation.",
            "Safety counts are material-only proxies, not WS3 policy verdict errors.",
            "Conservative fusion is a predeclared heuristic, not learned from test labels.",
            "Two CNN backbones require additional runtime and model storage.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Selected ConvNeXt weight (meta only):", chosen)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
