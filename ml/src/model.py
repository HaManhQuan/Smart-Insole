"""
model.py — LSTM Architecture
==============================
Input  : (batch, 100, 4)  — 2 giây @ 50Hz, 4 sensor
Output : (batch, 3)       — softmax: [Normal, Parkinson, Abnormal]
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Hằng số — khớp với preprocess.py
# ---------------------------------------------------------------------------

WINDOW_SIZE  = 100   # timesteps
N_SENSORS    = 4     # S1 S2 S3 S4
N_CLASSES    = 3     # Normal / Parkinson / Abnormal
CLASS_NAMES  = ["Normal", "Parkinson", "Abnormal"]


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_lstm_model(
    window_size: int        = WINDOW_SIZE,
    n_sensors: int          = N_SENSORS,
    n_classes: int          = N_CLASSES,
    lstm_units: list[int]   = [64, 32],
    dropout_rate: float     = 0.4,
    l2_lambda: float        = 1e-4,
    learning_rate: float    = 1e-3,
) -> Model:
    """
    Xây dựng LSTM model 2 lớp với:
      - Bidirectional LSTM lớp 1 (capture pattern cả 2 chiều thời gian)
      - LSTM lớp 2 (cô đọng đặc trưng)
      - BatchNormalization sau mỗi lớp LSTM
      - Dropout để regularize
      - Dense 16 → Dense 3 (softmax)

    Tại sao Bidirectional ở lớp 1:
      Gait pattern có context 2 chiều — bước tiếp theo ảnh hưởng
      cách đọc bước hiện tại (đặc biệt với Parkinson freezing of gait).
      Lớp 2 không cần Bi vì đã học đủ context từ lớp 1.
    """
    inputs = layers.Input(shape=(window_size, n_sensors), name="sensor_input")

    # Lớp 1: Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units[0],
            return_sequences=True,          # cần True để lớp LSTM sau nhận sequence
            kernel_regularizer=regularizers.l2(l2_lambda),
            recurrent_regularizer=regularizers.l2(l2_lambda),
            name="lstm_1",
        ),
        name="bi_lstm_1",
    )(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_rate, name="drop_1")(x)

    # Lớp 2: LSTM thường — cô đọng
    x = layers.LSTM(
        lstm_units[1],
        return_sequences=False,             # False vì đây là lớp cuối recurrent
        kernel_regularizer=regularizers.l2(l2_lambda),
        recurrent_regularizer=regularizers.l2(l2_lambda),
        name="lstm_2",
    )(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(dropout_rate, name="drop_2")(x)

    # Dense head
    x = layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_lambda),
        name="dense_head",
    )(x)
    x = layers.Dropout(dropout_rate / 2, name="drop_head")(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="smart_insole_lstm")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )

    return model


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# Thêm import numpy nếu chưa có
import numpy as np

class TestEvalCallback(tf.keras.callbacks.Callback):
    """Eval test set sau mỗi epoch — chỉ forward pass, không ảnh hưởng training."""
    def __init__(self, X_test: np.ndarray, y_test: np.ndarray):
        super().__init__()
        self.X_test  = X_test
        self.y_test  = y_test
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        y_proba = self.model.predict(self.X_test, verbose=0)
        y_pred  = np.argmax(y_proba, axis=1)
        acc     = float(np.mean(y_pred == self.y_test))
        self.history.append({"epoch": epoch + 1, "test_acc": acc})
        print(f"  → Test acc epoch {epoch + 1}: {acc:.4f}")


def build_callbacks(
    checkpoint_path: Path,
    log_dir: Optional[Path] = None,
    patience_early_stop: int = 15,
    patience_lr: int = 7,
    min_lr: float = 1e-6,
    X_test: Optional[np.ndarray] = None,   # THÊM
    y_test: Optional[np.ndarray] = None,   # THÊM
) -> list:
    """
    Trả về danh sách callback chuẩn cho training.

    EarlyStopping     : dừng nếu val_loss không giảm sau patience epoch
    ModelCheckpoint   : chỉ lưu khi val_loss tốt hơn (best only)
    ReduceLROnPlateau : giảm LR khi val_loss plateau — tránh stuck
    TensorBoard       : log metrics nếu có log_dir
    TestEvalCallback  : snapshot test acc từng epoch (không ảnh hưởng weights)
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    cbs = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1,
        ),
    ]

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        cbs.append(TensorBoard(log_dir=str(log_dir), histogram_freq=1))

    if X_test is not None and y_test is not None:   # THÊM
        cbs.append(TestEvalCallback(X_test, y_test))

    return cbs


# ---------------------------------------------------------------------------
# Tính class weights — xử lý imbalance
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray, n_classes: int = N_CLASSES) -> dict:
    """
    Tính class weight tỉ lệ nghịch với tần suất xuất hiện.
    Truyền vào train.py dưới dạng class_weight argument của model.fit().

    Ví dụ: Normal 1000 windows, Parkinson 600, Abnormal 200
    → Abnormal sẽ có weight cao hơn → model không ignore class nhỏ.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    return weight_dict


# ---------------------------------------------------------------------------
# Load model đã train — dùng trong inference.py
# ---------------------------------------------------------------------------

def load_trained_model(model_path: Path) -> Model:
    """
    Load model từ file .h5 hoặc SavedModel directory.
    Raise FileNotFoundError nếu không tìm thấy.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    return model


# ---------------------------------------------------------------------------
# Kiểm tra nhanh khi chạy trực tiếp
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Kiểm tra model ===\n")

    model = build_lstm_model()
    model.summary()

    # Kiểm tra forward pass với batch giả
    dummy = np.random.rand(8, WINDOW_SIZE, N_SENSORS).astype(np.float32)
    out = model(dummy, training=False)

    print(f"\nInput shape : {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output sum  : {out.numpy().sum(axis=1)}")   # phải xấp xỉ 1.0 (softmax)

    # Kiểm tra class weights
    y_fake = np.array([0]*100 + [1]*60 + [2]*20)
    weights = compute_class_weights(y_fake)
    print(f"\nClass weights (imbalanced data): {weights}")

    print("\n=== OK ===")