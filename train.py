"""
Training & Evaluation Pipeline for the Medical Cancer Detection CNN.
Performs:
  1. Dataset verification (auto-generates synthetic histopathology dataset if dataset folder is empty)
  2. Image loading, OpenCV preprocessing (Gaussian denoise + CLAHE), and [0, 1] normalization
  3. Data augmentation and preview generation
  4. TensorFlow/Keras CNN training (with automatic fallback to NumpyConvFeatureClassifier if TensorFlow is absent)
  5. Comprehensive evaluation on the held-out test set (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
  6. Saving of diagnostic Matplotlib figures to `outputs/`
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from generate_sample_dataset import build_sample_dataset
from preprocessing import load_dataset_split, visualize_preprocessing_stages
from augmentation import create_augmented_dataset, get_keras_image_data_generator, save_augmentation_preview
from model import TF_AVAILABLE, build_cancer_detection_cnn, get_training_callbacks, NumpyConvFeatureClassifier


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Computes binary classification metrics (Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC,
    and Confusion Matrix) using pure NumPy.
    """
    y_pred = (y_prob >= threshold).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = max(len(y_true), 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Compute ROC curve & AUC via trapezoidal rule
    thresholds = np.linspace(1.0, 0.0, 101)
    tpr_list = []
    fpr_list = []
    pos_total = max(int(np.sum(y_true == 1)), 1)
    neg_total = max(int(np.sum(y_true == 0)), 1)

    for thresh in thresholds:
        pred_t = y_prob >= thresh
        tpr_list.append(float(np.sum((y_true == 1) & pred_t)) / pos_total)
        fpr_list.append(float(np.sum((y_true == 0) & pred_t)) / neg_total)

    auc_val = float(np.trapezoid(tpr_list, fpr_list)) if hasattr(np, "trapezoid") else float(np.trapz(tpr_list, fpr_list))

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(abs(auc_val), 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "roc_curve": {"fpr": fpr_list, "tpr": tpr_list},
    }


def plot_training_history(history_dict: dict, save_path: Path) -> None:
    """
    Plots Training vs. Validation Loss and Accuracy curves over epochs.
    """
    epochs_range = range(1, len(history_dict["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), dpi=160)
    fig.suptitle("CNN Training & Validation Learning Curves", fontsize=13, fontweight="bold")

    axes[0].plot(epochs_range, history_dict["loss"], "o-", color="#1f77b4", label="Training Loss", linewidth=2, markersize=4)
    axes[0].plot(epochs_range, history_dict["val_loss"], "s--", color="#d62728", label="Validation Loss", linewidth=2, markersize=4)
    axes[0].set_title("Binary Cross-Entropy Loss", fontsize=11)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, history_dict["accuracy"], "o-", color="#2ca02c", label="Training Accuracy", linewidth=2, markersize=4)
    axes[1].plot(epochs_range, history_dict["val_accuracy"], "s--", color="#ff7f0e", label="Validation Accuracy", linewidth=2, markersize=4)
    axes[1].set_title("Classification Accuracy", fontsize=11)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.4, 1.03)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_and_roc(metrics: dict, save_path: Path) -> None:
    """
    Renders side-by-side Confusion Matrix heatmap and ROC-AUC curve on the test split.
    """
    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)
    fig.suptitle("Test Set Evaluation: Confusion Matrix & ROC Curve", fontsize=13, fontweight="bold")

    im = axes[0].imshow(matrix, cmap="Blues")
    axes[0].set_title("Confusion Matrix (Test Set)", fontsize=11)
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Non-Cancerous (0)", "Cancerous (1)"])
    axes[0].set_yticklabels(["Non-Cancerous (0)", "Cancerous (1)"])
    axes[0].set_xlabel("Predicted Class", fontweight="bold")
    axes[0].set_ylabel("True Class", fontweight="bold")

    thresh = matrix.max() / 2.0
    for i in range(2):
        for j in range(2):
            axes[0].text(
                j,
                i,
                f"{matrix[i, j]}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white" if matrix[i, j] > thresh else "#112233",
            )
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    fpr = metrics["roc_curve"]["fpr"]
    tpr = metrics["roc_curve"]["tpr"]
    axes[1].plot(fpr, tpr, color="#d62728", linewidth=2.5, label=f"CNN Classifier (AUC = {metrics['roc_auc']:.4f})")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline (AUC = 0.5000)")
    axes[1].fill_between(fpr, tpr, alpha=0.12, color="#d62728")
    axes[1].set_title("Receiver Operating Characteristic (ROC)", fontsize=11)
    axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[1].set_ylabel("True Positive Rate (Sensitivity / Recall)")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_training_pipeline():
    # 1. Ensure dataset exists
    if not (config.TRAIN_DIR / "cancerous").exists():
        print("[INFO] No dataset found in dataset/. Generating synthetic histopathology dataset...")
        build_sample_dataset()

    # 2. Load and preprocess train, val, and test splits
    print("[INFO] Loading and preprocessing medical images with OpenCV...")
    X_train, y_train, train_files = load_dataset_split(config.TRAIN_DIR)
    X_val, y_val, _ = load_dataset_split(config.VAL_DIR)
    X_test, y_test, test_files = load_dataset_split(config.TEST_DIR)
    print(f"  - Train shape: {X_train.shape} | Val shape: {X_val.shape} | Test shape: {X_test.shape}")

    # 3. Save preprocessing and augmentation diagnostic plots
    visualize_preprocessing_stages(
        Path(train_files[0]),
        config.OUTPUT_DIR / "preprocessing_pipeline.png",
    )
    save_augmentation_preview(
        X_train[0],
        config.OUTPUT_DIR / "data_augmentation_samples.png",
    )

    # 4. Expand training set via data augmentation
    X_train_aug, y_train_aug = create_augmented_dataset(X_train, y_train, multiplier=2)
    print(f"[INFO] Augmented training dataset size: {X_train_aug.shape[0]} samples")

    # 5. Train CNN model
    if TF_AVAILABLE:
        print("[INFO] Building and training TensorFlow/Keras CNN model...")
        model = build_cancer_detection_cnn()
        model.summary()
        datagen = get_keras_image_data_generator()
        history_obj = model.fit(
            datagen.flow(X_train_aug, y_train_aug, batch_size=config.BATCH_SIZE),
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            callbacks=get_training_callbacks(config.MODEL_SAVE_PATH),
            verbose=1,
        )
        history_dict = history_obj.history
        model.save(config.MODEL_SAVE_PATH)
        y_prob_test = model.predict(X_test, verbose=0).ravel()
    else:
        print("[INFO] TensorFlow binary not found in current environment; running NumPy+OpenCV convolutional classifier...")
        fallback_model = NumpyConvFeatureClassifier(learning_rate=0.08, epochs=config.EPOCHS)
        history_dict = fallback_model.fit(X_train_aug, y_train_aug, X_val, y_val)
        fallback_model.save(config.MODEL_DIR / "cancer_detection_weights.npz")
        y_prob_test = fallback_model.predict_proba(X_test)

    # 6. Compute evaluation metrics and save plots
    metrics = compute_binary_metrics(y_test, y_prob_test, threshold=config.CLASSIFICATION_THRESHOLD)
    plot_training_history(history_dict, config.OUTPUT_DIR / "training_history.png")
    plot_confusion_matrix_and_roc(metrics, config.OUTPUT_DIR / "confusion_matrix_and_roc.png")

    metrics_report = {
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall (Sensitivity)": metrics["recall"],
        "Specificity": metrics["specificity"],
        "F1-Score": metrics["f1_score"],
        "ROC-AUC": metrics["roc_auc"],
        "Confusion Matrix": metrics["confusion_matrix"],
    }
    with open(config.OUTPUT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    print("[INFO] Test Set Evaluation Metrics:")
    for k, v in metrics_report.items():
        print(f"  - {k}: {v}")

    return metrics_report, test_files, y_test, y_prob_test


if __name__ == "__main__":
    run_training_pipeline()
