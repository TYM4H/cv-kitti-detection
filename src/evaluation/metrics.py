"""
Вычисление mAP для torchvision моделей через torchmetrics.
Ultralytics-модели считают метрики сами через model.val().
"""

import time

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def compute_metrics(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Возвращает полный словарь метрик (как у Ultralytics):
    mAP50, mAP50_95, precision, recall, per_class, speed_ms.
    """
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    model.eval()

    inference_times = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]

            t0 = time.perf_counter()
            preds = model(images)
            inference_times.append((time.perf_counter() - t0) / len(images) * 1000)

            preds_cpu = [
                {
                    "boxes": p["boxes"].cpu(),
                    "scores": p["scores"].cpu(),
                    "labels": p["labels"].cpu(),
                }
                for p in preds
            ]
            targets_cpu = [
                {"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()}
                for t in targets
            ]
            metric.update(preds_cpu, targets_cpu)

    r = metric.compute()

    map_per_class = r.get("map_per_class", [None] * 3)
    mar_per_class = r.get("mar_100_per_class", [None] * 3)

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "mAP50": round(float(map_per_class[i]), 4) if map_per_class[i] is not None else None,
            "mAR": round(float(mar_per_class[i]), 4) if mar_per_class[i] is not None else None,
        }

    avg_inference = round(sum(inference_times) / len(inference_times), 2) if inference_times else 0.0

    return {
        "mAP50": round(float(r["map_50"]), 4),
        "mAP50_95": round(float(r["map"]), 4),
        "recall": round(float(r.get("mar_100", 0)), 4),
        "per_class": per_class,
        "speed_ms": {"inference": avg_inference},
    }
