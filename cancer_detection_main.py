"""
Cancer Detection from Medical Images using Deep Learning (CNN)
==============================================================
Tech Stack: Python, TensorFlow/Keras, OpenCV (cv2), NumPy, Matplotlib

Complete end-to-end script covering all internship submission requirements:
  1. Medical Image Preprocessing (OpenCV resizing, Gaussian denoising, CLAHE contrast enhancement)
  2. Pixel Intensity Normalization to [0.0, 1.0]
  3. Data Augmentation (rotations, flips, zoom crops, stain/contrast jitter, Keras ImageDataGenerator)
  4. Deep Convolutional Neural Network (CNN) in TensorFlow/Keras for binary classification
  5. Prediction of input medical images into Cancerous (Malignant) or Non-Cancerous (Benign) classes
"""

from pathlib import Path
import train
import predict
import config


def main():
    print("=" * 72)
    print("  CANCER DETECTION FROM MEDICAL IMAGES - CNN PIPELINE")
    print("=" * 72)
    metrics, test_files, y_test, _ = train.run_training_pipeline()

    # Generate 6-sample test prediction grid and a single image inference demo
    benign_indices = [i for i, y in enumerate(y_test) if y == 0][:3]
    malignant_indices = [i for i, y in enumerate(y_test) if y == 1][:3]
    selected_indices = benign_indices + malignant_indices
    sample_paths = [test_files[i] for i in selected_indices]
    sample_labels = [y_test[i] for i in selected_indices]

    predict.generate_sample_predictions_grid(
        sample_paths,
        sample_labels,
        config.OUTPUT_DIR / "sample_predictions.png",
    )

    single_demo = predict.predict_single_image(
        Path(sample_paths[-1]),
        config.OUTPUT_DIR / "single_image_inference.png",
    )
    print("\n[INFO] Single Image Inference Demo:")
    for k, v in single_demo.items():
        print(f"  - {k}: {v}")
    print(f"\n[SUCCESS] All visual plots and metrics saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
