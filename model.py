"""
Convolutional Neural Network (CNN) Architecture for Medical Cancer Detection.
Built with TensorFlow / Keras (`tensorflow.keras`).
Also includes a lightweight NumPy-compatible fallback trainer/evaluator so the codebase
can still execute and generate diagnostic metrics in restricted environments where
the TensorFlow binary is not installed.
"""

from typing import Tuple
import numpy as np
import config

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def build_cancer_detection_cnn(
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    learning_rate: float = config.LEARNING_RATE,
    dropout_rate: float = config.DROPOUT_RATE,
):
    """
    Builds and compiles a 4-block Deep Convolutional Neural Network (CNN) in TensorFlow/Keras
    for binary cancer classification (Cancerous vs. Non-Cancerous).

    Architecture:
      - Block 1: Conv2D(32, 3x3) -> BatchNorm -> ReLU -> Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
      - Block 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.25)
      - Block 3: Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.3)
      - Block 4: Conv2D(256, 3x3) -> BatchNorm -> ReLU -> GlobalAveragePooling2D
      - Classifier Head: Dense(128, ReLU) -> BatchNorm -> Dropout(0.4) -> Dense(1, Sigmoid)
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required to build the Keras CNN model. "
            "Install dependencies via: pip install -r requirements.txt"
        )

    inputs = layers.Input(shape=input_shape, name="medical_image_input")

    # Convolutional Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.20, name="drop1")(x)

    # Convolutional Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Convolutional Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.30, name="drop3")(x)

    # Convolutional Block 4 (last conv layer used for Grad-CAM localization)
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv4_last")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)

    # Global feature aggregation & dense classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, use_bias=False, name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.Activation("relu", name="relu_fc1")(x)
    x = layers.Dropout(dropout_rate, name="drop_fc1")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="cancer_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CancerDetectionCNN")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def get_training_callbacks(model_save_path=config.MODEL_SAVE_PATH):
    """
    Returns standard Keras training callbacks: ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau.
    """
    if not TF_AVAILABLE:
        return []

    return [
        callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


class NumpyConvFeatureClassifier:
    """
    Self-contained NumPy + OpenCV multi-filter convolutional feature extractor & logistic neural classifier.
    Used automatically if TensorFlow is not installed in a lightweight test container, ensuring
    deterministic end-to-end training, weight serialization, and inference verification.
    """

    def __init__(self, learning_rate: float = 0.08, epochs: int = config.EPOCHS):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.feat_mean = None
        self.feat_std = None

    @staticmethod
    def extract_conv_bank_features(X: np.ndarray) -> np.ndarray:
        """
        Applies a bank of 2D convolutional kernels (Gabor, Laplacian, Sobel, and Stain-Space pooling)
        via OpenCV to extract a 24-dimensional feature vector per image.
        """
        import cv2

        kernels = [
            np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),  # High-freq nuclear edge kernel
            np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),      # Horizontal Sobel
            np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),      # Vertical Sobel
            cv2.getGaborKernel((7, 7), 2.0, 0.0, 5.0, 0.5, 0, ktype=cv2.CV_32F),
            cv2.getGaborKernel((7, 7), 2.0, np.pi / 2, 5.0, 0.5, 0, ktype=cv2.CV_32F),
        ]

        feats = []
        for img in X:
            row = []
            # Per-channel mean, std, and quartiles
            for c in range(3):
                ch = img[:, :, c]
                row.extend([
                    float(np.mean(ch)),
                    float(np.std(ch)),
                    float(np.percentile(ch, 15)),
                    float(np.percentile(ch, 85)),
                ])
            # Convolutional filter responses on grayscale optical density
            gray = np.mean(img, axis=2).astype(np.float32)
            # Dark nuclear pixel ratio (hematoxylin density)
            dark_ratio = float(np.mean(gray < 0.52))
            very_dark_ratio = float(np.mean(gray < 0.38))
            row.extend([dark_ratio, very_dark_ratio])

            for k in kernels:
                fmap = np.maximum(0.0, cv2.filter2D(gray, cv2.CV_32F, k))
                row.append(float(np.mean(fmap)))
                row.append(float(np.std(fmap)))
            feats.append(row)
        return np.array(feats, dtype=np.float32)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        F_train = self.extract_conv_bank_features(X_train)
        F_val = self.extract_conv_bank_features(X_val)

        self.feat_mean = F_train.mean(axis=0, keepdims=True)
        self.feat_std = F_train.std(axis=0, keepdims=True) + 1e-6

        Z_train = (F_train - self.feat_mean) / self.feat_std
        Z_val = (F_val - self.feat_mean) / self.feat_std

        n_samples, n_features = Z_train.shape
        rng = np.random.default_rng(config.RANDOM_SEED)
        self.weights = rng.normal(0, 0.05, size=(n_features,)).astype(np.float32)
        self.bias = 0.0

        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        eps = 1e-7

        for _ in range(self.epochs):
            # Mini-batch SGD
            perm = rng.permutation(n_samples)
            for start in range(0, n_samples, config.BATCH_SIZE):
                idx = perm[start : start + config.BATCH_SIZE]
                xb = Z_train[idx]
                yb = y_train[idx].astype(np.float32)
                preds = self._sigmoid(xb @ self.weights + self.bias)
                err = preds - yb
                grad_w = (xb.T @ err) / len(xb) + 1e-3 * self.weights
                grad_b = float(np.mean(err))
                self.weights -= self.lr * grad_w
                self.bias -= self.lr * grad_b

            train_p = np.clip(self._sigmoid(Z_train @ self.weights + self.bias), eps, 1 - eps)
            val_p = np.clip(self._sigmoid(Z_val @ self.weights + self.bias), eps, 1 - eps)

            train_loss = -float(np.mean(y_train * np.log(train_p) + (1 - y_train) * np.log(1 - train_p)))
            val_loss = -float(np.mean(y_val * np.log(val_p) + (1 - y_val) * np.log(1 - val_p)))
            train_acc = float(np.mean((train_p >= 0.5).astype(np.int32) == y_train))
            val_acc = float(np.mean((val_p >= 0.5).astype(np.int32) == y_val))

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        F = self.extract_conv_bank_features(X)
        Z = (F - self.feat_mean) / self.feat_std
        return self._sigmoid(Z @ self.weights + self.bias)

    def save(self, path) -> None:
        np.savez(
            str(path),
            weights=self.weights,
            bias=np.array([self.bias], dtype=np.float32),
            feat_mean=self.feat_mean,
            feat_std=self.feat_std,
        )

    @classmethod
    def load(cls, path) -> "NumpyConvFeatureClassifier":
        data = np.load(str(path))
        obj = cls()
        obj.weights = data["weights"]
        obj.bias = float(data["bias"][0])
        obj.feat_mean = data["feat_mean"]
        obj.feat_std = data["feat_std"]
        return obj
