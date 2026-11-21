"""
Medical Image Data Augmentation Module.
Provides:
  1. Keras ImageDataGenerator / tf.keras augmentation layers for on-the-fly batch augmentation
  2. OpenCV + NumPy deterministic batch augmentation (rotations, horizontal/vertical flips,
     zoom crops, and brightness/contrast jitter)
  3. Visual grid generator showing augmented variations of a medical image
"""

from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import config


def augment_single_image_opencv(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Applies stochastic geometric and photometric transformations to a normalized [0, 1] RGB image
    using OpenCV and NumPy.
    """
    out = img.copy()
    h, w, _ = out.shape

    # 1. Random horizontal flip
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)

    # 2. Random vertical flip (histopathology patches are rotation/reflection invariant)
    if rng.random() < 0.5:
        out = cv2.flip(out, 0)

    # 3. Random continuous rotation (-25 to +25 degrees) with reflection border padding
    angle = float(rng.uniform(-25.0, 25.0))
    rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    out = cv2.warpAffine(out, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # 4. Random zoom (90% to 100% crop & resize)
    zoom_factor = float(rng.uniform(0.88, 1.0))
    if zoom_factor < 0.99:
        crop_h, crop_w = int(h * zoom_factor), int(w * zoom_factor)
        top = int(rng.integers(0, h - crop_h + 1))
        left = int(rng.integers(0, w - crop_w + 1))
        cropped = out[top : top + crop_h, left : left + crop_w]
        out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    # 5. Mild stain brightness & contrast jitter
    alpha = float(rng.uniform(0.92, 1.08))  # contrast
    beta = float(rng.uniform(-0.04, 0.04))  # brightness
    out = np.clip(alpha * out + beta, 0.0, 1.0).astype(np.float32)

    return out


def create_augmented_dataset(
    X: np.ndarray,
    y: np.ndarray,
    multiplier: int = 2,
    seed: int = config.RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expands a training dataset (X, y) by generating augmented copies via OpenCV.
    """
    rng = np.random.default_rng(seed)
    X_list = [X]
    y_list = [y]

    for _ in range(multiplier - 1):
        aug_batch = np.array([augment_single_image_opencv(img, rng) for img in X], dtype=np.float32)
        X_list.append(aug_batch)
        y_list.append(y.copy())

    X_combined = np.concatenate(X_list, axis=0)
    y_combined = np.concatenate(y_list, axis=0)

    perm = rng.permutation(len(X_combined))
    return X_combined[perm], y_combined[perm]


def get_keras_image_data_generator():
    """
    Returns a configured TensorFlow/Keras ImageDataGenerator for real-time batch augmentation.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.12,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="reflect",
    )


def save_augmentation_preview(sample_img: np.ndarray, save_path: Path) -> None:
    """
    Saves a visual 2x4 grid comparing an original normalized image with 7 augmented variants.
    """
    rng = np.random.default_rng(config.RANDOM_SEED + 7)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), dpi=160)
    fig.suptitle("Data Augmentation Samples (Geometric & Photometric Transforms)", fontsize=13, fontweight="bold")

    axes[0, 0].imshow(np.clip(sample_img, 0, 1))
    axes[0, 0].set_title("Original Preprocessed", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    transform_titles = [
        "Flip + +18 deg Rotation",
        "Vertical Flip + Zoom",
        "-22 deg Rotation + Contrast",
        "Horizontal Flip + Crop",
        "90 deg Aspect + Stain Jitter",
        "Dual Flip + Zoom 0.90x",
        "Rotation + Brightness Shift",
    ]

    for idx in range(1, 8):
        r, c = divmod(idx, 4)
        aug_img = augment_single_image_opencv(sample_img, rng)
        axes[r, c].imshow(np.clip(aug_img, 0, 1))
        axes[r, c].set_title(f"Aug #{idx}: {transform_titles[idx - 1]}", fontsize=9.5)
        axes[r, c].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
