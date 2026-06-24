"""
KITTIDataset для torchvision detection models (Faster R-CNN, RetinaNet).
YOLO-модели используют свой DataLoader из ultralytics; этот класс нужен только
для torchvision, где targets передаются как список словарей.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
CLASS_MAP = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}  # 0 = background


class KITTIDetectionDataset(Dataset):
    """Читает YOLO-разметку (из convert_kitti.py) и отдаёт torchvision-формат."""

    def __init__(self, images_dir: str, labels_dir: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms

        self.image_files = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes, labels = [], []

        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                # YOLO normalised centre -> absolute xyxy
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id + 1)  # shift: 0 reserved for background

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)

        return img, target


def get_transform(train: bool):
    augs = [T.ToTensor()]
    if train:
        augs.insert(0, T.RandomHorizontalFlip(0.5))
    return T.Compose(augs)


def collate_fn(batch):
    """torchvision требует список тензоров, а не стек — боксы разной длины."""
    return tuple(zip(*batch))
