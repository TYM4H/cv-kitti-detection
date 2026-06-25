# HANDOFF: CV-KITTI-Detection — написание отчёта
Копируй этот файл целиком в начало нового чата.

---

## Контекст

Учебная практика БВТ. Задача — детекция объектов на датасете KITTI (Car, Pedestrian, Cyclist).
Все эксперименты завершены. Нужно написать отчёт по структуре методички.

Репозиторий: https://github.com/TYM4H/cv-kitti-detection
Методичка: Учебная практика.pdf (структура отчёта описана ниже)

---

## Структура отчёта (по методичке)

1. **Введение** — предметная область (автономное вождение), актуальность, цель
2. **Обзор литературы** — не менее 15 источников (см. список ниже)
3. **Постановка задачи** — формализация, входные данные, метрики (mAP50, P, R)
4. **Набор данных** — KITTI, структура, EDA, распределение классов
5. **Методы и модели** — 5 архитектур, описание, гиперпараметры, обоснование выбора
6. **Эксперименты** — обучение + эксперименты с гиперпараметрами RT-DETR
7. **Результаты** — таблицы + графики
8. **Обсуждение** — анализ различий, сильные/слабые стороны
9. **Заключение** — лучшая модель, направления дальнейших исследований

---

## Датасет KITTI

- 7481 изображений (train split: 5985 train / 1496 val, seed=42)
- 3 класса: Car (83% объектов), Pedestrian (12%), Cyclist (5%)
- Размер изображений: 1242×375 px
- Всего аннотаций: 34 856 объектов
- Сильный дисбаланс классов — это важно упомянуть в обсуждении

---

## Результаты всех 5 моделей (основной эксперимент, 10 эпох)

| Модель | mAP50 | mAP50-95 | P | R | Inference ms | Params |
|--------|-------|----------|---|---|--------------|--------|
| RT-DETR-L | **0.896** | **0.625** | 0.878 | 0.826 | 39.0 | 32M |
| Faster R-CNN | 0.757 | 0.456 | — | 0.574 | 73.4 | 41M |
| RetinaNet | 0.719 | 0.431 | — | 0.554 | 67.6 | 34M |
| YOLOv8n | 0.691 | 0.434 | 0.778 | 0.612 | 0.7 | 3.0M |
| YOLOv10n | 0.589 | 0.376 | 0.679 | 0.519 | 1.0 | 2.3M |

### По классам (RT-DETR baseline):
| Класс | mAP50 | P | R |
|-------|-------|---|---|
| Car | 0.973 | 0.925 | 0.936 |
| Pedestrian | 0.832 | 0.850 | 0.741 |
| Cyclist | 0.883 | 0.858 | 0.802 |

### По классам (YOLOv8n):
| Класс | mAP50 | P | R |
|-------|-------|---|---|
| Car | 0.882 | 0.865 | 0.798 |
| Pedestrian | 0.594 | 0.722 | 0.508 |
| Cyclist | 0.598 | 0.743 | 0.525 |

---

## Эксперименты с гиперпараметрами RT-DETR

Все эксперименты на RT-DETR-L (лучшая модель).

| Эксперимент | mAP50 | mAP50-95 | P | R | Inference ms |
|-------------|-------|----------|---|---|--------------|
| baseline (e=10, sz=640, lr=auto) | 0.896 | 0.625 | 0.878 | 0.826 | 39.0 |
| imgsz=1280 (e=10, sz=1280) | **0.903** | **0.642** | 0.894 | 0.855 | 164.5 |
| lr=0.0001 (e=10) | ждём результата | | | | |
| lr=0.01 (e=10) | ждём результата | | | | |
| epochs=15 | ждём результата | | | | |

Последние 3 эксперимента запущены на ночь. После их завершения нужно:
- подтянуть `git pull` в репо
- дополнить таблицу
- построить график зависимости mAP50 от lr

---

## Ключевые выводы для отчёта

1. **RT-DETR лидирует** — transformer-based детектор превзошёл все остальные на 14+ пунктов mAP50
2. **YOLOv10 проиграл YOLOv8 по mAP, но postprocess в 5 раз быстрее** (0.3ms vs 1.6ms) за счёт NMS-free архитектуры
3. **Дисбаланс классов** — Pedestrian и Cyclist стабильно хуже Car у всех моделей. RetinaNet с Focal Loss не дал ожидаемого прироста на миноритарных классах
4. **imgsz=1280 даёт +0.7 mAP50 при 4× медленнее inference** — трейдофф точность/скорость
5. **Faster R-CNN и RetinaNet проиграли RT-DETR** несмотря на большее число параметров — двухстадийность не помогла при ограниченном числе эпох
6. **YOLOv8n — лучший выбор для реального времени**: 0.7ms inference при 0.691 mAP50

---

## Список литературы (15 источников)

Уже есть (5):
1. Jocher et al. "Ultralytics YOLOv8" (2023) — https://github.com/ultralytics/ultralytics
2. Wang et al. "YOLOv10: Real-Time End-to-End Object Detection" arXiv:2405.14458 (2024)
3. Ren et al. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" NeurIPS 2015
4. Lin et al. "Focal Loss for Dense Object Detection" (RetinaNet) ICCV 2017
5. Lv et al. "DETRs Beat YOLOs on Real-time Object Detection" (RT-DETR) arXiv:2304.08069 (2023)

Нужно добавить ещё 10:
6. Geiger et al. "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite" CVPR 2012 — описание датасета
7. Redmon et al. "You Only Look Once: Unified, Real-Time Object Detection" CVPR 2016 — оригинальный YOLO
8. Liu et al. "SSD: Single Shot MultiBox Detector" ECCV 2016
9. Lin et al. "Feature Pyramid Networks for Object Detection" CVPR 2017 — FPN backbone
10. Carion et al. "End-to-End Object Detection with Transformers" (DETR) ECCV 2020
11. He et al. "Deep Residual Learning for Image Recognition" (ResNet) CVPR 2016
12. Bochkovskiy et al. "YOLOv4: Optimal Speed and Accuracy of Object Detection" arXiv 2020
13. Zhu et al. "Deformable DETR" ICLR 2021
14. Girshick et al. "Rich Feature Hierarchies for Accurate Object Detection" (R-CNN) CVPR 2014
15. Everingham et al. "The Pascal Visual Object Classes Challenge" IJCV 2010 — метрика mAP

---

## Структура репозитория

```
cv-kitti-detection/
  configs/kitti.yaml              # конфиг датасета
  src/
    dataset/
      convert_kitti.py            # конвертер KITTI→YOLO
      dataset.py                  # KITTIDetectionDataset для torchvision
    models/
      yolo.py                     # YOLOv8n, YOLOv10n
      rtdetr.py                   # RT-DETR-L
      faster_rcnn.py              # Faster R-CNN ResNet50-FPN
      retinanet.py                # RetinaNet ResNet50-FPN
    training/train.py             # единый run() диспетчер
    evaluation/metrics.py         # compute_metrics() через torchmetrics
    utils/utils.py                # графики, таблицы
  notebooks/kaggle_training.ipynb # Kaggle ноутбук
  results/
    logs/                         # JSON с метриками каждой модели
    plots/                        # графики
  runs/detect/results/            # Ultralytics авто-графики (PR, confusion matrix)
  main.py                         # python main.py --model rtdetr
```

---

## Графики уже в репо

- `results/plots/class_distribution.png` — EDA распределение классов
- `results/plots/comparison.png` — bar chart mAP50 всех 5 моделей
- `results/plots/loss_faster_rcnn.png` — loss curve Faster R-CNN
- `results/plots/loss_retinanet.png` — loss curve RetinaNet
- `runs/detect/results/{model}/` — PR-кривые, confusion matrix, val predictions для YOLOv8n, YOLOv10n, RT-DETR

Для отчёта нужно ещё построить:
- График зависимости mAP50 от lr (после ночных экспериментов)
- Сравнительные кривые по классам

---

## Технические детали

- Платформа: Kaggle, GPU Tesla T4 (15GB)
- torch 2.10.0+cu128, ultralytics 8.4.75
- KITTI конвертер: только Car/Pedestrian/Cyclist, остальные классы игнорируются
- Faster R-CNN / RetinaNet: torchvision, кастомный collate_fn, class 0 = background
- Все YOLO/RT-DETR эксперименты через Ultralytics API
