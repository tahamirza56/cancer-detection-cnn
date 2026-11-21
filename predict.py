"""
Inference & Visual Prediction Module for Cancer Detection.
Loads an input medical image, applies OpenCV preprocessing and normalization,
predicts whether the tissue belongs to the Cancerous (Malignant) or Non-Cancerous (Benign) class,
and generates a spatial nuclear/lesion attention overlay using OpenCV and Matplotlib.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from preprocessing import preprocess_image
from model import TF_AVAILABLE, NumpyConvFeatureClassifier


def compute_anomaly_attention_map(normalized_rgb: np.ndarray) -> np.ndarray:
    """
    Computes a spatial nuclear/chromatin density saliency heatmap in [0, 1]
    highlighting hyperchromatic cellular clusters in the tissue sample.
    """
    inv_intensity = 1.0 - np.mean(normalized_rgb, axis=2).astype(np.float32)
    lap = np.abs(cv2.Laplacian(inv_intensity, cv2.CV_32F, ksize=3))
    combined = 0.70 * np.clip((inv_intensity - 0.30) / 0.50, 0, 1) + 0.30 * np.clip(lap * 2.5, 0, 1)
    smoothed = cv2.GaussianBlur(combined, (15, 15), 4.0)
    min_v, max_v = float(smoothed.min()), float(smoothed.max())
    if max_v - min_v > 1e-6:
        smoothed = (smoothed - min_v) / (max_v - min_v)
    return smoothed


def load_trained_predictor():
    """
    Loads the trained Keras CNN model (if present) or the saved NumPy convolutional classifier weights.
    """
    if TF_AVAILABLE and config.MODEL_SAVE_PATH.exists():
        import tensorflow as tf
        keras_model = tf.keras.models.load_model(str(config.MODEL_SAVE_PATH))
        return ("keras", keras_model)

    npz_path = config.MODEL_DIR / "cancer_detection_weights.npz"
    if npz_path.exists():
        return ("numpy", NumpyConvFeatureClassifier.load(npz_path))

    raise FileNotFoundError(
        "No trained model weights found. Run `python train.py` first to train and save the model."
    )


def predict_single_image(image_path: Path, save_plot_path: Path = None) -> dict:
    """
    Predicts whether a single input medical image is Cancerous or Non-Cancerous.
    """
    backend, predictor = load_trained_predictor()
    preprocessed = preprocess_image(image_path, normalize=True)
    batch = np.expand_dims(preprocessed, axis=0)

    if backend == "keras":
        prob_cancer = float(predictor.predict(batch, verbose=0).ravel()[0])
    else:
        prob_cancer = float(predictor.predict_proba(batch).ravel()[0])

    pred_class = 1 if prob_cancer >= config.CLASSIFICATION_THRESHOLD else 0
    pred_label = config.CLASS_LABELS[pred_class]
    confidence = prob_cancer if pred_class == 1 else (1.0 - prob_cancer)

    result = {
        "image_path": str(image_path),
        "predicted_class_index": pred_class,
        "predicted_label": pred_label,
        "cancer_probability": round(prob_cancer, 4),
        "confidence_percent": round(confidence * 100.0, 2),
    }

    if save_plot_path is not None:
        heatmap = compute_anomaly_attention_map(preprocessed)
        heatmap_u8 = np.uint8(255 * heatmap)
        jet_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay = np.clip(0.62 * preprocessed + 0.38 * jet_rgb, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=160)
        badge_color = "#d62728" if pred_class == 1 else "#2ca02c"
        fig.suptitle(
            f"Prediction: {pred_label} | Confidence: {result['confidence_percent']:.1f}% (P(Cancer) = {prob_cancer:.3f})",
            fontsize=12.5,
            fontweight="bold",
            color=badge_color,
        )

        axes[0].imshow(preprocessed)
        axes[0].set_title("Preprocessed Input Image", fontsize=10)
        axes[0].axis("off")

        im = axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Cellular Density Attention Map", fontsize=10)
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        axes[2].set_title("Diagnostic Region Overlay", fontsize=10)
        axes[2].axis("off")

        plt.tight_layout()
        fig.savefig(save_plot_path, bbox_inches="tight")
        plt.close(fig)

    return result


def generate_sample_predictions_grid(image_paths, true_labels, save_path: Path) -> None:
    """
    Creates a 2x3 diagnostic grid of test set predictions with ground-truth vs. predicted class labels.
    """
    backend, predictor = load_trained_predictor()
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), dpi=160)
    fig.suptitle("Sample Test Set Predictions (Cancerous vs. Non-Cancerous)", fontsize=13.5, fontweight="bold")

    for idx, (img_path, y_true) in enumerate(zip(image_paths[:6], true_labels[:6])):
        r, c = divmod(idx, 3)
        preprocessed = preprocess_image(img_path, normalize=True)
        batch = np.expand_dims(preprocessed, axis=0)
        if backend == "keras":
            prob = float(predictor.predict(batch, verbose=0).ravel()[0])
        else:
            prob = float(predictor.predict_proba(batch).ravel()[0])

        pred_cls = 1 if prob >= config.CLASSIFICATION_THRESHOLD else 0
        conf = prob if pred_cls == 1 else (1.0 - prob)
        is_correct = pred_cls == int(y_true)
        title_color = "#1b7837" if is_correct else "#c51b7d"

        axes[r, c].imshow(preprocessed)
        axes[r, c].set_title(
            f"True: {config.CLASS_LABELS[int(y_true)]}\n"
            f"Pred: {config.CLASS_LABELS[pred_cls]} ({conf*100:.1f}%)",
            fontsize=9.5,
            fontweight="bold",
            color=title_color,
        )
        axes[r, c].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a medical image into Cancerous or Non-Cancerous.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input medical image")
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.OUTPUT_DIR / "single_prediction.png"),
        help="Path to save the annotated prediction plot",
    )
    args = parser.parse_args()

    res = predict_single_image(Path(args.image), Path(args.output))
    print("[INFO] Prediction Result:")
    for k, v in res.items():
        print(f"  - {k}: {v}")
