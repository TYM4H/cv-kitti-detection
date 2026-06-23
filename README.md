# CV-KITTI-Detection

Сравнение 5 моделей детекции объектов на датасете KITTI (интеллектуальные
транспортные системы). Учебная практика.

## Модели
| Модель | Тип | Библиотека |
|---|---|---|
| YOLOv8n | одностадийная, anchor-free | Ultralytics |
| YOLOv10n | одностадийная, anchor-free, NMS-free | Ultralytics |
| RT-DETR | трансформер | Ultralytics |
| Faster R-CNN | двустадийная | torchvision |
| RetinaNet | одностадийная, anchor-based | torchvision |

## Запуск на Kaggle
```bash
git clone https://github.com/ТВОЙ_НИК/cv-kitti-detection.git
cd cv-kitti-detection
pip install -r requirements.txt -q

# 1. Конвертация аннотаций KITTI -> YOLO
python src/dataset/convert_kitti.py \
    --images /kaggle/input/kitti/training/image_2 \
    --labels /kaggle/input/kitti/training/label_2 \
    --out    /kaggle/working/kitti_yolo

# 2. Обучение (пример для YOLOv8)
python main.py --model yolov8n
```

## Структура
- `configs/`   — конфиги экспериментов и датасета
- `data/`      — данные (в гит не коммитятся, см .gitignore)
- `src/dataset/`    — загрузка и конвертация данных
- `src/models/`     — фабрика torchvision-моделей
- `src/training/`   — циклы обучения
- `src/evaluation/` — подсчёт метрик (mAP, Precision, Recall)
- `src/utils/`      — seed, логирование, визуализация
- `results/`   — графики и логи прогонов

## Воспроизводимость
`set_seed(42)` везде. Версии зафиксированы в `requirements.txt`.
