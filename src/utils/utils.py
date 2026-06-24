"""
Визуализация и вспомогательные функции.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
COLORS = ["#2196F3", "#FF5722", "#4CAF50"]


def plot_class_distribution(labels_dir: str, save_path: str) -> None:
    """Гистограмма распределения классов по train-сплиту."""
    from collections import Counter
    counter: Counter = Counter()
    for f in Path(labels_dir).glob("*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                counter[int(line.split()[0])] += 1

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(
        [CLASS_NAMES[i] for i in sorted(counter)],
        [counter[i] for i in sorted(counter)],
        color=COLORS,
    )
    plt.title("Распределение классов в KITTI (train)")
    plt.ylabel("Количество объектов")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {save_path}")


def plot_comparison(logs_dir: str = "results/logs", save_path: str = "results/plots/comparison.png") -> None:
    """Сравнительный bar-chart mAP50 по всем обученным моделям."""
    logs_dir = Path(logs_dir)
    models, map50s = [], []
    for f in sorted(logs_dir.glob("*.json")):
        data = json.loads(f.read_text())
        models.append(data["model"])
        map50s.append(data["mAP50"])

    if not models:
        print("Нет логов для сравнения")
        return

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(models))
    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, map50s, color="#5C6BC0")
    plt.xticks(x, models, rotation=15, ha="right")
    plt.ylabel("mAP@0.5")
    plt.title("Сравнение моделей на KITTI")
    plt.ylim(0, 1)
    for bar, val in zip(bars, map50s):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Сохранено: {save_path}")


def plot_train_losses(log_path: str, save_path: str) -> None:
    """Кривая обучения по epochs для torchvision-моделей."""
    data = json.loads(Path(log_path).read_text())
    losses = data.get("train_losses", [])
    if not losses:
        print("train_losses не найдены в логе")
        return

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss — {data['model']}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Сохранено: {save_path}")


def print_results_table(logs_dir: str = "results/logs") -> None:
    """Печатает сводную таблицу метрик в консоль."""
    header = f"{'Модель':<35} {'mAP50':>7} {'mAP50-95':>9} {'P':>7} {'R':>7}"
    print(header)
    print("-" * len(header))
    for f in sorted(Path(logs_dir).glob("*.json")):
        d = json.loads(f.read_text())
        p = d.get('precision', None)
        r = d.get('recall', None)
        p_str = f"{p:>7.4f}" if isinstance(p, float) else f"{'—':>7}"
        r_str = f"{r:>7.4f}" if isinstance(r, float) else f"{'—':>7}"
        print(f"{d['model']:<35} {d.get('mAP50', 0):>7.4f} "
              f"{d.get('mAP50_95', 0):>9.4f} {p_str} {r_str}")
