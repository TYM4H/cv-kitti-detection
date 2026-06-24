"""
Вычисление mAP для torchvision моделей через torchmetrics.
Ultralytics-модели считают метрики сами через model.val().
"""

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def compute_map(
    model,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
) -> tuple[float, float]:
    """Возвращает (mAP@IoU=0.5, mAP@IoU=0.5:0.95)."""
    metric = MeanAveragePrecision(iou_type="bbox")
    model.eval()

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            preds = model(images)

            # torchmetrics ожидает CPU-тензоры
            preds_cpu = [
                {
                    "boxes": p["boxes"].cpu(),
                    "scores": p["scores"].cpu(),
                    "labels": p["labels"].cpu(),
                }
                for p in preds
            ]
            targets_cpu = [
                {
                    "boxes": t["boxes"].cpu(),
                    "labels": t["labels"].cpu(),
                }
                for t in targets
            ]
            metric.update(preds_cpu, targets_cpu)

    result = metric.compute()
    map50 = float(result["map_50"])
    map50_95 = float(result["map"])
    return map50, map50_95
