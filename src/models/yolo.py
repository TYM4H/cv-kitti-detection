"""
Обёртка для YOLO-моделей через Ultralytics API (YOLOv8n, YOLOv10n).
"""

import json
from pathlib import Path

from ultralytics import YOLO


def train_yolo(
    model_name: str,
    data_cfg: str,
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 64,
    device: int = 0,
    project: str = "results",
) -> dict:
    """Обучает модель, возвращает словарь метрик."""
    model = YOLO(f"{model_name}.pt")
    model.train(
        data=data_cfg,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=model_name,
        exist_ok=True,
    )
    metrics = model.val()
    result = {
        "model": model_name,
        "epochs": epochs,
        "batch": batch,
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
    _save_metrics(result, project, model_name)
    return result


def _save_metrics(result: dict, project: str, name: str) -> None:
    log_dir = Path(project) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Метрики сохранены: {log_dir / name}.json")
