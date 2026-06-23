"""
Конвертер аннотаций KITTI -> YOLO формат.

KITTI хранит один .txt на изображение. Каждая строка - один объект:
    Car 0.00 0 -1.57 614.24 181.78 727.31 284.77 1.57 1.62 3.89 ...
Нам нужны только:
    [0]  имя класса        -> Car / Pedestrian / Cyclist (остальное выкидываем)
    [4]  x1 (left)         \
    [5]  y1 (top)           |  bounding box в АБСОЛЮТНЫХ пикселях
    [6]  x2 (right)         |
    [7]  y2 (bottom)       /

YOLO формат (одна строка на объект):
    <class_id> <cx> <cy> <w> <h>
где все четыре координаты НОРМАЛИЗОВАНЫ на ширину/высоту картинки (0..1),
а cx,cy - это ЦЕНТР бокса, а не угол. Вот тут все и спотыкаются.

Запуск:
    python src/dataset/convert_kitti.py \
        --images /kaggle/input/kitti/training/image_2 \
        --labels /kaggle/input/kitti/training/label_2 \
        --out    /kaggle/working/kitti_yolo \
        --val-split 0.2
"""

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

# Три класса по плану. KITTI содержит больше (Van, Truck, Tram, Misc,
# DontCare), но методичка и наш kitti.yaml завязаны на три. Всё что не тут -
# молча игнорируем. Это осознанное решение, не баг: на защите так и говори.
CLASS_MAP = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
}


def convert_one_label(label_path: Path, img_w: int, img_h: int) -> list[str]:
    """Один KITTI .txt -> список строк в YOLO формате."""
    yolo_lines: list[str] = []
    for raw in label_path.read_text().splitlines():
        parts = raw.split()
        if len(parts) < 8:
            continue
        cls_name = parts[0]
        if cls_name not in CLASS_MAP:
            continue  # Van, Truck, DontCare и прочее - мимо

        cls_id = CLASS_MAP[cls_name]
        x1, y1, x2, y2 = map(float, parts[4:8])

        # абсолютные углы -> нормализованный центр + размеры
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        # защита от мусорных аннотаций, вылезающих за кадр
        if w <= 0 or h <= 0:
            continue
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        w, h = min(w, 1.0), min(h, 1.0)

        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return yolo_lines


def build_dataset(images_dir, labels_dir, out_dir, val_split=0.2, seed=42):
    images_dir, labels_dir, out_dir = map(Path, (images_dir, labels_dir, out_dir))

    # целевая структура под Ultralytics: images/{train,val} + labels/{train,val}
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    image_files = sorted(p for p in images_dir.iterdir()
                         if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not image_files:
        raise FileNotFoundError(f"Ни одной картинки в {images_dir} - проверь путь")

    random.seed(seed)  # тот самый seed=42, воспроизводимость как в чеклисте
    random.shuffle(image_files)
    n_val = int(len(image_files) * val_split)
    val_set = set(image_files[:n_val])

    stats = {"train": 0, "val": 0, "skipped_no_label": 0, "objects": 0}

    for img_path in image_files:
        split = "val" if img_path in val_set else "train"
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            stats["skipped_no_label"] += 1
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        yolo_lines = convert_one_label(label_path, img_w, img_h)

        shutil.copy(img_path, out_dir / "images" / split / img_path.name)
        (out_dir / "labels" / split / f"{img_path.stem}.txt").write_text(
            "\n".join(yolo_lines)
        )
        stats[split] += 1
        stats["objects"] += len(yolo_lines)

    print("=" * 48)
    print(f"  Готово. Train: {stats['train']}, Val: {stats['val']}")
    print(f"  Всего объектов размечено: {stats['objects']}")
    print(f"  Пропущено (нет .txt): {stats['skipped_no_label']}")
    print(f"  Результат в: {out_dir}")
    print("=" * 48)
    return stats


def parse_args():
    ap = argparse.ArgumentParser(description="KITTI -> YOLO converter")
    ap.add_argument("--images", required=True, help="папка с image_2")
    ap.add_argument("--labels", required=True, help="папка с label_2")
    ap.add_argument("--out", required=True, help="куда складывать YOLO-датасет")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(args.images, args.labels, args.out, args.val_split, args.seed)
