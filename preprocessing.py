"""
Medical Image Preprocessing & Normalization Module.
Uses OpenCV (cv2) and NumPy to perform:
  1. Image loading and BGR -> RGB conversion
  2. Spatial resizing to standard CNN input dimensions (128x128)
  3. Gaussian denoising to suppress high-frequency scanner noise
  4. Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel (LAB color space)
     to normalize stain illumination and highlight cellular/nuclear morphology
  5. Min-Max / Floating-point normalization to [0.0, 1.0]
"""

from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np
import matplotlib.pyplot as plt
import config


def apply_clahe_rgb(rgb_img: np.ndarray) -> np.ndarray:
    """
    Enhances local contrast of an RGB medical image using CLAHE in the CIE LAB color space.
    """
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_GRID_SIZE,
    )
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def preprocess_image(
    image_path_or_array,
    target_size: Tuple[int, int] = (config.IMG_WIDTH, config.IMG_HEIGHT),
    apply_clahe: bool = config.APPLY_CLAHE,
    normalize: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single medical image.
    Accepts either a file path (str/Path) or an in-memory BGR/RGB NumPy array.
    Returns a preprocessed RGB image normalized to [0.0, 1.0] (if normalize=True).
    """
    if isinstance(image_path_or_array, (str, Path)):
        bgr = cv2.imread(str(image_path_or_array))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image at: {image_path_or_array}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path_or_array, np.ndarray):
        rgb = image_path_or_array.copy()
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
    else:
        raise TypeError("Input must be a file path or NumPy ndarray.")

    # Step 1: Resize image to target resolution
    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)

    # Step 2: Gaussian denoising to reduce sensor/compression artifacts
    denoised = cv2.GaussianBlur(resized, config.GAUSSIAN_BLUR_KERNEL, 0)

    # Step 3: Local contrast enhancement via CLAHE
    enhanced = apply_clahe_rgb(denoised) if apply_clahe else denoised

    # Step 4: Intensity normalization to [0.0, 1.0]
    if normalize:
        normalized = enhanced.astype(np.float32) / 255.0
        return normalized

    return enhanced


def load_dataset_split(split_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads all images from a split directory structured as:
        split_dir/
            non_cancerous/ (or 0/)
            cancerous/     (or 1/)
    Returns:
        X: float32 array of shape (N, H, W, C) normalized to [0, 1]
        y: int32 array of shape (N,) with labels 0 or 1
        filepaths: list of image file paths
    """
    images = []
    labels = []
    filepaths = []

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    # Support either ["non_cancerous", "cancerous"] or ["0", "1"] or ["benign", "malignant"]
    folder_mappings = [
        (0, ["non_cancerous", "0", "benign", "normal"]),
        (1, ["cancerous", "1", "malignant", "tumor"]),
    ]

    for class_idx, candidate_names in folder_mappings:
        for name in candidate_names:
            class_folder = split_dir / name
            if class_folder.exists() and class_folder.is_dir():
                for img_path in sorted(class_folder.rglob("*")):
                    if img_path.suffix.lower() in valid_exts:
                        img_norm = preprocess_image(img_path, normalize=True)
                        images.append(img_norm)
                        labels.append(class_idx)
                        filepaths.append(str(img_path))
                break

    if not images:
        raise RuntimeError(f"No valid images found in {split_dir}")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # Shuffle deterministically
    rng = np.random.default_rng(config.RANDOM_SEED)
    indices = rng.permutation(len(X))
    return X[indices], y[indices], [filepaths[i] for i in indices]


def visualize_preprocessing_stages(image_path: Path, save_path: Path) -> None:
    """
    Generates a 4-stage comparison figure showing Raw, Denoised, CLAHE-Enhanced,
    and Normalized channel intensity distribution for a medical image.
    """
    bgr = cv2.imread(str(image_path))
    rgb_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_raw, (config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    denoised = cv2.GaussianBlur(resized, config.GAUSSIAN_BLUR_KERNEL, 0)
    clahe_img = apply_clahe_rgb(denoised)
    normalized = clahe_img.astype(np.float32) / 255.0

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), dpi=160)
    fig.suptitle("Medical Image Preprocessing & Normalization Pipeline (OpenCV)", fontsize=13, fontweight="bold")

    axes[0].imshow(resized)
    axes[0].set_title("1. Original Resized (128x128)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(denoised)
    axes[1].set_title("2. Gaussian Denoised (3x3)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(clahe_img)
    axes[2].set_title("3. CLAHE Contrast Enhanced", fontsize=10)
    axes[2].axis("off")

    colors = ("#d62728", "#2ca02c", "#1f77b4")
    channel_names = ("Red Channel", "Green Channel", "Blue Channel")
    for c, (col, cname) in enumerate(zip(colors, channel_names)):
        vals = normalized[:, :, c].ravel()
        axes[3].hist(vals, bins=32, alpha=0.55, color=col, label=cname, density=True)
    axes[3].set_title("4. Normalized Pixel Distribution [0, 1]", fontsize=10)
    axes[3].set_xlabel("Normalized Intensity")
    axes[3].set_ylabel("Density")
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
