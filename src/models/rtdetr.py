"""
RT-DETR через Ultralytics API.
Трансформер-детектор без NMS (аналогично YOLOv10).
batch=8 — трансформер потребляет много памяти на T4.
"""

import json
from pathlib import Path

from ultralytics import RTDETR


def train_rtdetr(
    data_cfg: str,
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 8,
    lr: float | None = None,
    device: int = 0,
    project: str = "results",
    name: str = "rtdetr",
) -> dict:
    model_name = "rtdetr-l"
    model = RTDETR(f"{model_name}.pt")
    train_kwargs = dict(
        data=data_cfg,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
    )
    if lr is not None:
        train_kwargs["lr0"] = lr
        train_kwargs["lrf"] = lr  # конечный lr = начальный (без decay)
    model.train(**train_kwargs)
    metrics = model.val()
    result = {
        "model": "rtdetr-l",
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "per_class": {
            name: {
                "mAP50": round(float(metrics.box.ap50[i]), 4),
                "P": round(float(metrics.box.p[i]), 4),
                "R": round(float(metrics.box.r[i]), 4),
            }
            for i, name in enumerate(["Car", "Pedestrian", "Cyclist"])
        },
        "speed_ms": {
            "preprocess": round(metrics.speed["preprocess"], 2),
            "inference": round(metrics.speed["inference"], 2),
            "postprocess": round(metrics.speed["postprocess"], 2),
        },
    }
    log_dir = Path(project) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Метрики RT-DETR сохранены: {log_dir}/{name}.json")
    return result
