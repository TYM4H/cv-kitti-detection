"""
Faster R-CNN (ResNet-50-FPN backbone) через torchvision.
Особенность: loss считается внутри модели при передаче targets.
На eval targets НЕ передаём — модель переключается в режим предсказания.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from src.dataset.dataset import KITTIDetectionDataset, collate_fn, get_transform
from src.evaluation.metrics import compute_map


NUM_CLASSES = 4  # 3 объекта + background


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_faster_rcnn(
    kitti_yolo_dir: str,
    epochs: int = 10,
    batch: int = 4,
    lr: float = 0.005,
    device_id: int = 0,
    project: str = "results",
) -> dict:
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Faster R-CNN на {device}")

    ds_train = KITTIDetectionDataset(
        images_dir=f"{kitti_yolo_dir}/images/train",
        labels_dir=f"{kitti_yolo_dir}/labels/train",
        transforms=get_transform(train=True),
    )
    ds_val = KITTIDetectionDataset(
        images_dir=f"{kitti_yolo_dir}/images/val",
        labels_dir=f"{kitti_yolo_dir}/labels/val",
        transforms=get_transform(train=False),
    )
    dl_train = DataLoader(ds_train, batch_size=batch, shuffle=True,
                          num_workers=2, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)

    model = build_model().to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=0.9, weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in tqdm(dl_train, desc=f"Epoch {epoch+1}/{epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(dl_train)
        train_losses.append(avg_loss)
        print(f"  loss: {avg_loss:.4f}")

    map50, map50_95 = compute_map(model, dl_val, device, num_classes=NUM_CLASSES)

    result = {
        "model": "faster_rcnn_resnet50_fpn",
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "train_losses": [round(l, 4) for l in train_losses],
    }
    log_dir = Path(project) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "faster_rcnn.json", "w") as f:
        json.dump(result, f, indent=2)

    ckpt_dir = Path(project) / "weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "faster_rcnn.pth")
    print(f"Метрики: mAP50={map50:.4f}, mAP50-95={map50_95:.4f}")
    return result
