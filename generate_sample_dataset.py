"""
Synthetic Histopathology Medical Image Dataset Generator.
Creates realistic H&E-stained tissue patch images for:
  - Class 0: non_cancerous (regular cellular structure, lighter pink/violet stroma, uniform nuclei)
  - Class 1: cancerous (hyperchromatic pleomorphic nuclei, dense irregular clusters, mitotic figures)

Allows the pipeline to run out-of-the-box for testing, grading, and demonstration
when an external Kaggle dataset (such as Breast Histopathology IDC or ISIC Skin Cancer)
is not yet mounted.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import config


def generate_histopathology_patch(is_cancerous: bool, rng: np.random.Generator, size: int = 128) -> np.ndarray:
    """
    Synthesizes an H&E-stained histopathological tissue patch in BGR format for OpenCV
    with realistic biological overlap and staining variability.
    """
    # Overlapping base stroma background (H&E eosin pink/lavender in BGR)
    if not is_cancerous:
        base_b = rng.integers(200, 235)
        base_g = rng.integers(170, 212)
        base_r = rng.integers(215, 246)
    else:
        base_b = rng.integers(188, 226)
        base_g = rng.integers(150, 196)
        base_r = rng.integers(202, 238)

    img = np.ones((size, size, 3), dtype=np.float32)
    img[:, :, 0] = base_b
    img[:, :, 1] = base_g
    img[:, :, 2] = base_r

    # Smooth stromal fiber texture using low-frequency sinusoidal waves
    y, x = np.mgrid[0:size, 0:size]
    angle = rng.uniform(0, np.pi)
    freq = rng.uniform(0.05, 0.14)
    wave = np.sin(freq * (x * np.cos(angle) + y * np.sin(angle)))
    for c in range(3):
        img[:, :, c] += wave * rng.uniform(6.0, 14.0)

    img = np.clip(img, 0, 255).astype(np.uint8)

    if not is_cancerous:
        # Benign: organized glandular rings + uniform stromal nuclei (with occasional dense reactive benign patch)
        num_glands = int(rng.integers(1, 4))
        for _ in range(num_glands):
            cx = int(rng.integers(24, size - 24))
            cy = int(rng.integers(24, size - 24))
            radius = int(rng.integers(13, 23))
            cv2.circle(img, (cx, cy), radius - 4, (240, 232, 246), -1, lineType=cv2.LINE_AA)
            num_ring_nuclei = int(rng.integers(9, 15))
            for k in range(num_ring_nuclei):
                theta = 2 * np.pi * k / num_ring_nuclei + rng.uniform(-0.12, 0.12)
                nx = int(cx + radius * np.cos(theta))
                ny = int(cy + radius * np.sin(theta))
                if 4 <= nx < size - 4 and 4 <= ny < size - 4:
                    cv2.circle(img, (nx, ny), int(rng.integers(2, 4)), (140, 70, 110), -1, lineType=cv2.LINE_AA)

        num_cells = int(rng.integers(22, 45))
        for _ in range(num_cells):
            cx = int(rng.integers(6, size - 6))
            cy = int(rng.integers(6, size - 6))
            r = int(rng.integers(2, 5))
            color = (int(rng.integers(120, 165)), int(rng.integers(50, 95)), int(rng.integers(90, 135)))
            cv2.circle(img, (cx, cy), r, color, -1, lineType=cv2.LINE_AA)
    else:
        # Malignant: invasive clusters of pleomorphic hyperchromatic nuclei
        num_clusters = int(rng.integers(1, 4))
        for _ in range(num_clusters):
            cluster_x = int(rng.integers(20, size - 20))
            cluster_y = int(rng.integers(20, size - 20))
            cv2.circle(
                img,
                (cluster_x, cluster_y),
                int(rng.integers(18, 34)),
                (int(rng.integers(170, 200)), int(rng.integers(125, 165)), int(rng.integers(180, 215))),
                -1,
                lineType=cv2.LINE_AA,
            )

            num_cells = int(rng.integers(20, 44))
            for _ in range(num_cells):
                cx = int(np.clip(rng.normal(cluster_x, 18), 5, size - 5))
                cy = int(np.clip(rng.normal(cluster_y, 18), 5, size - 5))
                ax1 = int(rng.integers(3, 7))
                ax2 = int(rng.integers(2, 6))
                rot = int(rng.integers(0, 180))
                color = (
                    int(rng.integers(90, 145)),
                    int(rng.integers(25, 75)),
                    int(rng.integers(60, 115)),
                )
                cv2.ellipse(img, (cx, cy), (ax1, ax2), rot, 0, 360, color, -1, lineType=cv2.LINE_AA)
                if ax1 >= 5:
                    cv2.circle(img, (cx, cy), 1, (55, 15, 40), -1)

    noise = rng.normal(0, 5.5, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.65)
    return img


def build_sample_dataset(
    base_dir: Path = config.DATASET_DIR,
    train_per_class: int = 120,
    val_per_class: int = 30,
    test_per_class: int = 30,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """
    Populates dataset/train, dataset/val, and dataset/test with synthetic medical images.
    """
    rng = np.random.default_rng(seed)
    splits = {
        "train": train_per_class,
        "val": val_per_class,
        "test": test_per_class,
    }
    counts = {}

    for split_name, count_per_class in splits.items():
        counts[split_name] = {}
        for class_idx, class_name in enumerate(config.CLASS_NAMES):
            class_dir = base_dir / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            is_cancerous = class_idx == 1
            for i in range(count_per_class):
                img = generate_histopathology_patch(
                    is_cancerous=is_cancerous,
                    rng=rng,
                    size=config.IMG_HEIGHT,
                )
                file_path = class_dir / f"{class_name}_{i:04d}.png"
                cv2.imwrite(str(file_path), img)
            counts[split_name][class_name] = count_per_class

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic medical histopathology dataset.")
    parser.add_argument("--train-per-class", type=int, default=120, help="Training images per class")
    parser.add_argument("--val-per-class", type=int, default=30, help="Validation images per class")
    parser.add_argument("--test-per-class", type=int, default=30, help="Test images per class")
    args = parser.parse_args()

    summary = build_sample_dataset(
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
    )
    print(f"[INFO] Sample dataset generated at: {config.DATASET_DIR}")
    for split, info in summary.items():
        print(f"  - {split}: {info}")
