# CV-KITTI-Detection

Сравнение пяти современных архитектур детекции объектов на датасете **KITTI**.  
Учебная практика БВТ — задача детекции Car / Pedestrian / Cyclist в сценах автономного вождения.

---

## Результаты

### 5 моделей (10 эпох, imgsz=640)

| Модель | mAP50 | mAP50-95 | P | R | Inference, мс | Params |
|--------|-------|----------|---|---|--------------|--------|
| **RT-DETR-L** | **0.896** | **0.625** | 0.878 | 0.826 | 39.0 | 32M |
| Faster R-CNN | 0.757 | 0.456 | — | 0.574 | 73.4 | 41M |
| RetinaNet | 0.719 | 0.431 | — | 0.554 | 67.6 | 34M |
| YOLOv8n | 0.691 | 0.434 | 0.778 | 0.612 | 0.7 | 3.0M |
| YOLOv10n | 0.589 | 0.376 | 0.679 | 0.519 | 1.0 | 2.3M |

### Эксперименты с гиперпараметрами

**RT-DETR-L:**

| Конфигурация | mAP50 | mAP50-95 | Inference, мс |
|---|---|---|---|
| baseline (imgsz=640) | 0.896 | 0.625 | 39.0 |
| imgsz=1280 | **0.903** | **0.642** | 164.5 |

**YOLOv8n:**

| Конфигурация | epochs | mAP50 | mAP50-95 | Inference, мс |
|---|---|---|---|---|
| baseline | 10 | 0.691 | 0.434 | 0.7 |
| lr=0.001 | 15* | 0.770 | 0.492 | 1.72 |
| imgsz=1280 | 10 | 0.816 | 0.554 | 4.62 |
| epochs=30 | 30 | **0.830** | **0.554** | 1.69 |

\* остановился по early stopping

---

## Структура проекта

```
cv-kitti-detection/
├── configs/
│   └── kitti.yaml              # конфиг датасета для Ultralytics
├── data/
│   ├── raw/                    # исходный датасет (не в git, см. .gitignore)
│   └── processed/              # конвертированные аннотации (не в git)
├── src/
│   ├── dataset/
│   │   ├── convert_kitti.py    # конвертер KITTI → YOLO формат
│   │   └── dataset.py          # KITTIDetectionDataset для torchvision
│   ├── models/
│   │   ├── yolo.py             # YOLOv8n, YOLOv10n (Ultralytics)
│   │   ├── rtdetr.py           # RT-DETR-L (Ultralytics)
│   │   ├── faster_rcnn.py      # Faster R-CNN ResNet50-FPN (torchvision)
│   │   └── retinanet.py        # RetinaNet ResNet50-FPN (torchvision)
│   ├── training/
│   │   └── train.py            # run() — единая точка запуска обучения
│   ├── evaluation/
│   │   └── metrics.py          # compute_metrics() через torchmetrics
│   └── utils/
│       └── utils.py            # графики, таблицы
├── notebooks/
│   ├── kaggle_training.ipynb   # обучение 5 базовых моделей
│   └── kaggle_experiments.ipynb # эксперименты с гиперпараметрами
├── results/
│   ├── logs/                   # JSON с метриками каждого эксперимента
│   └── plots/                  # графики (EDA, сравнение, loss curves)
├── runs/
│   └── detect/results/         # артефакты Ultralytics (PR-кривые, confusion matrix)
├── main.py                     # CLI: python main.py --model rtdetr
└── requirements.txt
```

---

## Быстрый старт (Kaggle)

```bash
git clone https://github.com/TYM4H/cv-kitti-detection.git
cd cv-kitti-detection
pip install -r requirements.txt -q
```

**1. Конвертация аннотаций KITTI → YOLO:**
```bash
python src/dataset/convert_kitti.py \
    --images /kaggle/input/klemenko/kitti-dataset/data_object_image_2/training/image_2 \
    --labels /kaggle/input/klemenko/kitti-dataset/data_object_label_2/training/label_2 \
    --out    /kaggle/working/kitti_yolo
```

**2. Обучение модели:**
```bash
python main.py --model rtdetr       # RT-DETR-L
python main.py --model yolov8n      # YOLOv8n
python main.py --model faster_rcnn  # Faster R-CNN
```

Или через API:
```python
from src.training.train import run

run('yolov8n', kitti_yolo_dir='/kaggle/working/kitti_yolo', epochs=30)
run('rtdetr',  kitti_yolo_dir='/kaggle/working/kitti_yolo', imgsz=1280)
```

---

## Датасет

[KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/) — сцены городского вождения.

- 7 481 изображение, разбиение 5 985 train / 1 496 val (seed=42)
- 3 класса: **Car** (82.5%) / **Pedestrian** (12%) / **Cyclist** (5.5%)
- 34 856 размеченных объектов

Датасет не входит в репозиторий. На Kaggle используется датасет `klemenko/kitti-dataset`.

---

## Платформа

- GPU: NVIDIA Tesla T4 (15 GB) — Kaggle Notebooks
- Python 3.10, PyTorch 2.10.0+cu128, Ultralytics 8.4.75, torchvision 0.15.2
- Воспроизводимость: `seed=42` зафиксирован везде
