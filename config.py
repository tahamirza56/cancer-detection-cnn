"""
Configuration module for the Cancer Detection CNN pipeline.
Defines dataset paths, image preprocessing parameters, augmentation settings,
and CNN training hyperparameters.
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Ensure directories exist
for directory in [MODEL_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Image dimensions & channels (compatible with Histopathology IDC / ISIC patches)
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Class mapping
CLASS_NAMES = ["non_cancerous", "cancerous"]
CLASS_LABELS = {
    0: "Non-Cancerous (Benign)",
    1: "Cancerous (Malignant)",
}

# Preprocessing settings
APPLY_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
GAUSSIAN_BLUR_KERNEL = (3, 3)

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.4
RANDOM_SEED = 42
CLASSIFICATION_THRESHOLD = 0.5

# Saved model file
MODEL_SAVE_PATH = MODEL_DIR / "cancer_detection_cnn.keras"
