"""
Точка входа. Пример:
    python main.py --model yolov8n
    python main.py --model faster_rcnn --epochs 10 --batch 4
    python main.py --model retinanet --lr 1e-4
    python main.py --compare   # распечатать таблицу уже обученных моделей
"""

import argparse

from src.training.train import run
from src.utils.utils import plot_comparison, print_results_table


def parse_args():
    p = argparse.ArgumentParser(description="KITTI Object Detection")
    p.add_argument("--model", type=str,
                   choices=["yolov8n", "yolov10n", "rtdetr", "faster_rcnn", "retinanet"],
                   help="Модель для обучения")
    p.add_argument("--data-cfg", default="configs/kitti.yaml")
    p.add_argument("--kitti-yolo-dir", default="/kaggle/working/kitti_yolo",
                   help="Путь к сконвертированному датасету")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--project", default="results")
    p.add_argument("--compare", action="store_true",
                   help="Показать таблицу + график по уже обученным моделям")
    return p.parse_args()


def main():
    args = parse_args()

    if args.compare:
        print_results_table(f"{args.project}/logs")
        plot_comparison(
            logs_dir=f"{args.project}/logs",
            save_path=f"{args.project}/plots/comparison.png",
        )
        return

    if not args.model:
        print("Укажи --model или --compare")
        return

    result = run(
        model=args.model,
        data_cfg=args.data_cfg,
        kitti_yolo_dir=args.kitti_yolo_dir,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        lr=args.lr,
    )
    print(f"\nmAP50: {result['mAP50']}  |  mAP50-95: {result['mAP50_95']}")


if __name__ == "__main__":
    main()
