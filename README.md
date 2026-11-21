# Medical Image Cancer Detection using CNN (TensorFlow/Keras & OpenCV)

An end-to-end deep learning pipeline that analyzes histopathological/medical images and classifies them into **Cancerous (Malignant)** or **Non-Cancerous (Benign)** categories using **Python, TensorFlow/Keras, OpenCV, NumPy, and Matplotlib**.

## 1. Project Overview & Features

- **Medical Image Preprocessing (`preprocessing.py`)**:
  - Loads medical images via OpenCV (`cv2`), converts BGR to RGB, and resizes to `128x128x3`.
  - Applies `3x3` Gaussian denoising to remove high-frequency scanner noise.
  - Enhances cellular and nuclear contrast using **CLAHE** (Contrast Limited Adaptive Histogram Equalization) in the CIE LAB color space.
- **Intensity Normalization (`preprocessing.py`)**:
  - Scales pixel intensities to floating-point `[0.0, 1.0]` values and visualizes RGB channel distributions.
- **Data Augmentation (`augmentation.py`)**:
  - Implements both real-time `tensorflow.keras.preprocessing.image.ImageDataGenerator` augmentation and OpenCV geometric/photometric transforms (random rotations, horizontal/vertical flips, zoom crops, and H&E stain brightness/contrast jitter).
- **Deep CNN Architecture (`model.py`)**:
  - 4-Block Convolutional Neural Network built with `tensorflow.keras` featuring `Conv2D`, `BatchNormalization`, `ReLU`, `MaxPooling2D`, `Dropout`, `GlobalAveragePooling2D`, and a `Sigmoid` binary classification head.
  - Includes `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau` callbacks.
- **Automated Evaluation & Single-Image Prediction (`train.py`, `predict.py`)**:
  - Outputs Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, ROC-AUC, and Confusion Matrix plots.
  - Classifies any input medical image into **Cancerous (Malignant)** or **Non-Cancerous (Benign)** with confidence percentage and a cellular density attention heatmap overlay.

## 2. Project Directory Structure

```text
cancer_detection_project/
├── README.md                      # Project documentation and usage guide
├── requirements.txt               # Python package dependencies
├── Cancer_Detection_CNN.ipynb     # Ready-to-run Jupyter / Google Colab Notebook
├── config.py                      # Hyperparameters, image dimensions, and paths
├── generate_sample_dataset.py     # Synthetic H&E histopathology dataset generator
├── preprocessing.py               # OpenCV CLAHE, denoising, and [0, 1] normalization
├── augmentation.py                # OpenCV & Keras ImageDataGenerator augmentation
├── model.py                       # TensorFlow/Keras 4-block CNN architecture
├── train.py                       # Training, validation, and test evaluation script
├── predict.py                     # Single-image and batch inference CLI tool
├── cancer_detection_main.py       # All-in-one entry point running the full pipeline
├── dataset/                       # Train / Val / Test medical image splits
│   ├── train/{non_cancerous,cancerous}/
│   ├── val/{non_cancerous,cancerous}/
│   └── test/{non_cancerous,cancerous}/
├── models/                        # Saved trained model weights
└── outputs/                       # Diagnostic Matplotlib charts & evaluation metrics
    ├── preprocessing_pipeline.png
    ├── data_augmentation_samples.png
    ├── training_history.png
    ├── confusion_matrix_and_roc.png
    ├── sample_predictions.png
    ├── single_image_inference.png
    └── evaluation_metrics.json
```

## 3. Installation & Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Full Pipeline (Dataset Setup + Preprocessing + CNN Training + Evaluation + Inference)**:
   ```bash
   python cancer_detection_main.py
   ```
   *(If `dataset/` is empty, the pipeline automatically generates a sample H&E histopathology dataset so you can run and verify the code immediately. To use an external Kaggle dataset such as Breast Histopathology Images (IDC) or ISIC Skin Cancer, place the class folders inside `dataset/train/`, `dataset/val/`, and `dataset/test/`.)*

3. **Predict on Any Single Medical Image**:
   ```bash
   python predict.py --image dataset/test/cancerous/cancerous_0000.png --output outputs/my_prediction.png
   ```

## 4. Using a Custom or Kaggle Dataset

The data loader in `preprocessing.py` automatically recognizes any of the following class subfolder names inside `dataset/train/`, `dataset/val/`, and `dataset/test/`:
- **Class 0 (Non-Cancerous)**: `non_cancerous/`, `0/`, `benign/`, or `normal/`
- **Class 1 (Cancerous)**: `cancerous/`, `1/`, `malignant/`, or `tumor/`
