"""Export Run 3 Swin-T predictions for an explicitly selected original data split.

    uv run --with torch --with torchvision --with timm python scripts/ws2_predict_swin_val.py

This performs inference only and never changes split assignments or trains.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from binbuddy_contracts.enums import Material
from binbuddy_ml.experiments import PROTOCOL_RUN3
from binbuddy_ml.experiments import model as model_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/ws2/models/run3/E2_public_plus_local_swin_t.pt"),
    )
    parser.add_argument(
        "--predictions", type=Path, default=Path("data/ws2/v0.2-local-data/predictions.csv")
    )
    parser.add_argument("--images", type=Path, default=Path("data/ws2/prepared_v2"))
    parser.add_argument("--output", type=Path, default=Path("data/ws2/run3/swin_validation.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--source", default=None, help="Optional exact dataset source, e.g. local")
    args = parser.parse_args()

    import timm
    import torch

    classes = [material.value for material in Material]
    spec = model_spec("swin_t")
    model = timm.create_model(
        spec.timm_id,
        pretrained=False,
        num_classes=len(classes),
        drop_rate=PROTOCOL_RUN3.drop_rate,
        drop_path_rate=PROTOCOL_RUN3.drop_path_rate,
    )
    # Only load a trusted checkpoint from the team; torch.load uses pickle.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    config = timm.data.resolve_data_config({}, model=model)
    size = PROTOCOL_RUN3.input_size
    transform = transforms.Compose(
        [
            transforms.Resize(int(size / 0.875)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"]),
        ]
    )
    with args.predictions.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["split"] == args.split
                and (args.source is None or row["source"] == args.source)]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No rows for split={args.split}, source={args.source}")
    output: list[dict[str, str]] = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe image path: {relative}")
            with Image.open(args.images / relative) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0)
            probabilities = torch.softmax(model(tensor).float(), dim=1)[0].numpy()
            record = {
                "path": row["path"],
                "source": row["source"],
                "split": row["split"],
                "material": row["material"],
                "group": row["group"],
                **{
                    f"p_{name}": f"{float(value):.8f}"
                    for name, value in zip(classes, probabilities, strict=True)
                },
                "predicted": classes[int(np.argmax(probabilities))],
            }
            output.append(record)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(rows)} {args.split} images", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(
        json.dumps(
            {"images": len(output), "split": args.split, "source": args.source, "output": str(args.output), "classes": classes}, indent=2
        )
    )


if __name__ == "__main__":
    main()
