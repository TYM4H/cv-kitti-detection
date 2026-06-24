"""
Единая точка запуска обучения.
Вызывается из main.py или напрямую из Kaggle-ноутбука:
    from src.training.train import run
    run("yolov8n", data_cfg="configs/kitti.yaml", kitti_yolo_dir="/kaggle/working/kitti_yolo")
"""

from src.models.yolo import train_yolo
from src.models.rtdetr import train_rtdetr
from src.models.faster_rcnn import train_faster_rcnn
from src.models.retinanet import train_retinanet

YOLO_MODELS = {"yolov8n", "yolov10n"}
RTDETR_MODELS = {"rtdetr", "rtdetr-l"}


def run(
    model: str,
    data_cfg: str = "configs/kitti.yaml",
    kitti_yolo_dir: str = "/kaggle/working/kitti_yolo",
    epochs: int = 10,
    batch: int | None = None,
    device: int = 0,
    project: str = "results",
    lr: float = 0.005,
) -> dict:
    model_lower = model.lower()

    if model_lower in YOLO_MODELS:
        return train_yolo(
            model_name=model_lower,
            data_cfg=data_cfg,
            epochs=epochs,
            batch=batch or 64,
            device=device,
            project=project,
        )
    elif model_lower in RTDETR_MODELS:
        return train_rtdetr(
            data_cfg=data_cfg,
            epochs=epochs,
            batch=batch or 8,
            device=device,
            project=project,
        )
    elif model_lower == "faster_rcnn":
        return train_faster_rcnn(
            kitti_yolo_dir=kitti_yolo_dir,
            epochs=epochs,
            batch=batch or 4,
            lr=lr,
            device_id=device,
            project=project,
        )
    elif model_lower == "retinanet":
        return train_retinanet(
            kitti_yolo_dir=kitti_yolo_dir,
            epochs=epochs,
            batch=batch or 4,
            lr=lr,
            device_id=device,
            project=project,
        )
    else:
        raise ValueError(
            f"Неизвестная модель: {model}. "
            f"Доступны: yolov8n, yolov10n, rtdetr, faster_rcnn, retinanet"
        )
