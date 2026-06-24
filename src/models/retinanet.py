"""
RetinaNet (ResNet-50-FPN) через torchvision.
Ключевое отличие от Faster R-CNN: Focal Loss в голове — решает дисбаланс классов.
Ожидаем лучший recall на Pedestrian и Cyclist по сравнению с YOLO.
API идентичен Faster R-CNN, но замена головы — через RetinaNetHead.
"""

import json
from pathlib import Path

import torchvision
if not hasattr(torchvision, '_is_tracing'):
    torchvision._is_tracing = lambda: False

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead
from tqdm import tqdm

from src.dataset.dataset import KITTIDetectionDataset, collate_fn, get_transform
from src.evaluation.metrics import compute_metrics

NUM_CLASSES = 4  # 3 + background (RetinaNet включает bg в num_classes)


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    model = retinanet_resnet50_fpn(pretrained=pretrained)
    in_channels = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head = RetinaNetHead(in_channels, num_anchors, num_classes)
    return model


def train_retinanet(
    kitti_yolo_dir: str,
    epochs: int = 10,
    batch: int = 4,
    lr: float = 1e-4,
    device_id: int = 0,
    project: str = "results",
) -> dict:
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"RetinaNet на {device}")

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
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

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

    metrics = compute_metrics(model, dl_val, device)

    result = {
        "model": "retinanet_resnet50_fpn",
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "train_losses": [round(l, 4) for l in train_losses],
        **metrics,
    }
    log_dir = Path(project) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "retinanet.json", "w") as f:
        json.dump(result, f, indent=2)

    ckpt_dir = Path(project) / "weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "retinanet.pth")
    print(f"Метрики: mAP50={metrics['mAP50']:.4f}, mAP50-95={metrics['mAP50_95']:.4f}")
    return result
