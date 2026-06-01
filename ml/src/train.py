"""
train.py — Training Script
===========================
Chạy:
    python ml/src/train.py

Yêu cầu:
    - ml/data/processed/ đã có X_train.npy, y_train.npy, X_val.npy, y_val.npy
      (chạy preprocess.py trước)
    - Thư mục ml/models/ sẽ được tạo tự động

Output:
    - ml/models/lstm_v1.h5          ← model tốt nhất (theo val_loss)
    - ml/models/training_log.json   ← history loss/accuracy mỗi epoch
    - ml/models/class_weights.json  ← class weights dùng lại khi fine-tune
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import logging
import time
import numpy as np
from pathlib import Path
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import sys
sys.path.insert(0, str(Path(__file__).parent))
from datetime import datetime
from model import (
    build_lstm_model,
    build_callbacks,
    compute_class_weights,
    CLASS_NAMES,
    N_CLASSES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cấu hình training
# ---------------------------------------------------------------------------

PROCESSED_DIR  = Path("ml/data/processed")
MODEL_DIR      = Path("ml/models")
from datetime import datetime
_RUN_TS        = datetime.now().strftime("%y%m%d%H%M")   # vd: 2605291054
RUN_DIR        = MODEL_DIR / "report" / _RUN_TS   # folder riêng mỗi lần train
MODEL_PATH     = MODEL_DIR / f"lstm_{_RUN_TS}.h5"
LOG_DIR        = MODEL_DIR / "tensorboard_logs"

EPOCHS         = 100     # EarlyStopping sẽ dừng sớm hơn thực tế
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3

# Hyperparameters — khớp với build_lstm_model()
LSTM_UNITS     = [64, 32]
DROPOUT_RATE   = 0.3
L2_LAMBDA      = 1e-4

# Callback patience
EARLY_STOP_PATIENCE = 15
LR_REDUCE_PATIENCE  = 7


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_processed_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load numpy arrays đã qua preprocess.py.
    Raise FileNotFoundError nếu chưa chạy preprocess.
    """
    required = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]
    for fname in required:
        path = PROCESSED_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {path}\n"
                "Hãy chạy preprocess.py trước: python ml/src/preprocess.py"
            )

    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(PROCESSED_DIR / "y_train.npy").astype(np.int8)
    X_val   = np.load(PROCESSED_DIR / "X_val.npy").astype(np.float32)
    y_val   = np.load(PROCESSED_DIR / "y_val.npy").astype(np.int8)

    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Log phân bố data
# ---------------------------------------------------------------------------

def log_data_summary(X_train, y_train, X_val, y_val) -> None:
    log.info("--- Dữ liệu training ---")
    log.info("X_train : %s  dtype=%s", X_train.shape, X_train.dtype)
    log.info("X_val   : %s  dtype=%s", X_val.shape,   X_val.dtype)

    for split_name, y in [("train", y_train), ("val", y_val)]:
        unique, counts = np.unique(y, return_counts=True)
        parts = [f"{CLASS_NAMES[u]}={c}" for u, c in zip(unique, counts)]
        log.info("%-5s classes: %s (total=%d)", split_name, " | ".join(parts), len(y))

    # Kiểm tra range
    log.info("X_train range: [%.4f, %.4f]", X_train.min(), X_train.max())
    log.info("X_val   range: [%.4f, %.4f]", X_val.min(),   X_val.max())


# ---------------------------------------------------------------------------
# Kiểm tra GPU
# ---------------------------------------------------------------------------

def log_device_info() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        log.info("GPU khả dụng: %s", [g.name for g in gpus])
        # Cho phép memory growth — tránh chiếm hết VRAM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        log.info("Không có GPU — training trên CPU (chậm hơn nhưng vẫn OK với dataset nhỏ)")


# ---------------------------------------------------------------------------
# Training chính
# ---------------------------------------------------------------------------

def train() -> dict:
    """
    Chạy toàn bộ training pipeline.
    Trả về history dict (loss, val_loss, accuracy, val_accuracy theo epoch).
    """
    log.info("=== Smart Insole — Training LSTM ===")
    log_device_info()

    # Load data
    X_train, y_train, X_val, y_val = load_processed_data()
    log_data_summary(X_train, y_train, X_val, y_val)

    # Class weights — xử lý imbalance
    class_weights = compute_class_weights(y_train, n_classes=N_CLASSES)
    log.info("Class weights: %s", {CLASS_NAMES[k]: f"{v:.3f}" for k, v in class_weights.items()})

    # Build model
    log.info("--- Build model ---")
    model = build_lstm_model(
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE,
        l2_lambda=L2_LAMBDA,
        learning_rate=LEARNING_RATE,
    )
    model.summary(print_fn=log.info)

    total_params = model.count_params()
    log.info("Tổng số parameters: %s", f"{total_params:,}")
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(np.int8)
    log.info("Test set loaded: %s windows", len(y_test))
    # Callbacks
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = build_callbacks(
        checkpoint_path=MODEL_PATH,
        log_dir=LOG_DIR,
        patience_early_stop=EARLY_STOP_PATIENCE,
        patience_lr=LR_REDUCE_PATIENCE,
        X_test=X_test,
        y_test=y_test,
    )

    # Training
    log.info("--- Bắt đầu training (tối đa %d epochs, batch=%d) ---", EPOCHS, BATCH_SIZE)
    t0 = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    elapsed = time.time() - t0
    log.info("Training xong — %.1f giây (%.1f phút)", elapsed, elapsed / 60)

    # Kết quả tốt nhất
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = min(history.history["val_loss"])
    best_val_acc  = history.history["val_accuracy"][best_epoch - 1]

    log.info("--- Kết quả tốt nhất ---")
    log.info("Epoch       : %d / %d", best_epoch, len(history.history["val_loss"]))
    log.info("Val loss    : %.4f", best_val_loss)
    log.info("Val accuracy: %.4f (%.1f%%)", best_val_acc, best_val_acc * 100)
    log.info("Model saved : %s", MODEL_PATH)

    # Lưu history ra JSON để notebook vẽ learning curve
    history_path = RUN_DIR / f"training_log_{_RUN_TS}.json"
    with open(history_path, "w") as f:
        # history.history chứa list[float] — JSON serializable
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f,
            indent=2,
        )
    log.info("History saved: %s", history_path)
    from model import TestEvalCallback
    test_cb = next((cb for cb in callbacks if isinstance(cb, TestEvalCallback)), None)
    if test_cb:
        test_acc_path = RUN_DIR / f"test_acc_per_epoch_{_RUN_TS}.json"
        with open(test_acc_path, "w") as f:
            json.dump(test_cb.history, f, indent=2)
        log.info("Test acc per epoch saved: %s", test_acc_path)
    # Lưu class weights để dùng lại khi fine-tune
    weights_path = RUN_DIR / f"class_weights_{_RUN_TS}.json"
    with open(weights_path, "w") as f:
        json.dump(class_weights, f, indent=2)
    log.info("Class weights saved: %s", weights_path)
    # ---------------------------------------------------------------------------
    # Vẽ và lưu các biểu đồ vào RUN_DIR
    # ---------------------------------------------------------------------------

    # 1. Learning Curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist      = history.history
    epochs_ran = range(1, len(hist["loss"]) + 1)
    best_epoch = int(np.argmin(hist["val_loss"])) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Learning Curve — Smart Insole LSTM", fontsize=13, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs_ran, hist["loss"],     "#534AB7", linewidth=1.8, label="Train loss")
    ax.plot(epochs_ran, hist["val_loss"], "#D85A30", linewidth=1.8, label="Val loss", linestyle="--")
    ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.2, label=f"Best epoch ({best_epoch})")
    ax.scatter([best_epoch], [hist["val_loss"][best_epoch - 1]], color="#D85A30", s=80, zorder=5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs_ran, hist["accuracy"],     "#1D9E75", linewidth=1.8, label="Train acc")
    ax.plot(epochs_ran, hist["val_accuracy"], "#B5A000", linewidth=1.8, label="Val acc", linestyle="--")
    ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.2, label=f"Best epoch ({best_epoch})")
    ax.scatter([best_epoch], [hist["val_accuracy"][best_epoch - 1]], color="#B5A000", s=80, zorder=5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    curve_path = RUN_DIR / f"learning_curve_{_RUN_TS}.png"
    plt.savefig(curve_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", curve_path)

    # 2. LR Schedule
    if "lr" in hist:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.semilogy(epochs_ran, hist["lr"], "#534AB7", linewidth=1.8)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate (log scale)")
        ax.set_title("Learning Rate Schedule")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        lr_path = RUN_DIR / f"lr_schedule_{_RUN_TS}.png"
        plt.savefig(lr_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved: %s", lr_path)

    # 3. Val Confusion Matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_proba_val  = model.predict(X_val, batch_size=64, verbose=0)
    y_pred_val   = np.argmax(y_proba_val, axis=1).astype(np.int8)
    present      = sorted(np.unique(np.concatenate([y_val, y_pred_val])).tolist())
    target_names = [CLASS_NAMES[i] for i in present]

    cm  = confusion_matrix(y_val, y_pred_val, labels=present)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names).plot(
        ax=ax, cmap="Blues", values_format="d", xticks_rotation=0
    )
    ax.set_title("Confusion Matrix — Val Set", fontsize=12, pad=10)
    plt.tight_layout()
    val_cm_path = RUN_DIR / f"val_confusion_matrix_{_RUN_TS}.png"
    plt.savefig(val_cm_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", val_cm_path)
    return history.history


# ---------------------------------------------------------------------------
# Fine-tune với data thực tế (dùng sau khi có bệnh nhân thực)
# ---------------------------------------------------------------------------

def fine_tune(
    X_new: np.ndarray,
    y_new: np.ndarray,
    base_model_path: Path = MODEL_PATH,
    output_path: Path     = MODEL_DIR / "lstm_v2_finetuned.h5",
    freeze_lstm: bool     = False,
    epochs: int           = 30,
    learning_rate: float  = 1e-4,     # LR nhỏ hơn training gốc để không overwrite
) -> None:
    """
    Fine-tune model đã train trên GaitPDB với data thực tế từ phòng khám.

    Tham số:
        X_new         : array (N, 100, 4) — data thực tế đã qua preprocess_realtime
        y_new         : array (N,) — label 0/1/2 do bác sĩ xác nhận
        freeze_lstm   : True = chỉ train Dense layers (nếu data thực tế ít < 50 windows)
                        False = train toàn bộ (nếu có > 200 windows)
        learning_rate : nên để nhỏ hơn training gốc ít nhất 10x

    Khi nào dùng:
        - Sau khi thu được data từ 20–30 bệnh nhân tại phòng khám
        - Label đã được bác sĩ xác nhận qua UPDRS/H&Y scale
        - Chạy: fine_tune(X_real, y_real)
    """
    from model import load_trained_model

    log.info("=== Fine-tuning từ %s ===", base_model_path)
    log.info("Data mới: %s windows | freeze_lstm=%s", X_new.shape[0], freeze_lstm)

    if X_new.shape[0] < 20:
        log.warning(
            "Chỉ có %d windows — quá ít để fine-tune đáng tin cậy. "
            "Cần ít nhất 20 windows (tương đương ~2 phút đi bộ).",
            X_new.shape[0],
        )

    model = load_trained_model(base_model_path)

    # Freeze LSTM layers nếu data ít
    if freeze_lstm:
        for layer in model.layers:
            if "lstm" in layer.name.lower() or "bi_lstm" in layer.name.lower():
                layer.trainable = False
                log.info("Frozen: %s", layer.name)

    # Compile lại với LR nhỏ hơn
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_weights = compute_class_weights(y_new, n_classes=N_CLASSES)

    callbacks = build_callbacks(
        checkpoint_path=output_path,
        patience_early_stop=10,
        patience_lr=5,
    )

    model.fit(
        X_new, y_new,
        validation_split=0.2,
        epochs=epochs,
        batch_size=16,               # batch nhỏ hơn vì data ít hơn
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    log.info("Fine-tuned model saved: %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def update_env_if_better(new_model_path: Path, new_val_loss: float) -> None:
    """
    So sánh val_loss của model vừa train với model đang được .env trỏ đến.
    Nếu model mới tốt hơn (val_loss thấp hơn) → tự cập nhật dòng MODEL_PATH trong .env.

    Cơ chế:
      - Đọc file .env ở thư mục gốc project (2 cấp trên ml/src/)
      - Tìm dòng MODEL_PATH=...
      - Load model cũ bằng TensorFlow để lấy val_loss thật (nếu có metrics.json thì dùng luôn)
      - So sánh và ghi đè nếu mới tốt hơn

    Không làm gì nếu:
      - Không tìm thấy file .env
      - Model cũ không tồn tại (lần train đầu tiên → tự động cập nhật)
      - val_loss mới >= val_loss cũ
    """
    # Tìm .env: thư mục gốc project = 2 cấp trên ml/src/
    project_root = Path(__file__).parents[2]
    env_path = project_root / ".env"

    if not env_path.exists():
        log.warning("Không tìm thấy .env tại %s — bỏ qua auto-update", env_path)
        return

    # Đọc .env, tìm MODEL_PATH hiện tại
    lines = env_path.read_text(encoding="utf-8").splitlines()
    current_model_str = None
    current_model_line_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("MODEL_PATH=") and not stripped.startswith("#"):
            current_model_str = stripped.split("=", 1)[1].strip()
            current_model_line_idx = i
            break

    if current_model_line_idx is None:
        log.warning(".env không có dòng MODEL_PATH — bỏ qua auto-update")
        return

    current_model_path = project_root / current_model_str

    # Lấy val_loss của model cũ
    # Ưu tiên đọc từ metrics.json gần nhất (nhanh, không cần load TF)
    old_val_loss = _read_best_val_loss(current_model_path)

    print("\n" + "─" * 50)
    if old_val_loss is None:
        # Lần đầu train, chưa có model cũ → dùng model mới luôn
        print(f"  [Auto-update] Chưa có model cũ → dùng model mới")
        _write_model_path_to_env(lines, current_model_line_idx, new_model_path, project_root, env_path)
    elif new_val_loss < old_val_loss:
        improvement = (old_val_loss - new_val_loss) / old_val_loss * 100
        print(f"  [Auto-update] Model mới TỐT HƠN ✓")
        print(f"    Val loss cũ : {old_val_loss:.4f}  ({current_model_path.name})")
        print(f"    Val loss mới: {new_val_loss:.4f}  ({new_model_path.name})")
        print(f"    Cải thiện   : -{improvement:.1f}%")
        _write_model_path_to_env(lines, current_model_line_idx, new_model_path, project_root, env_path)
    else:
        print(f"  [Auto-update] Model mới KHÔNG tốt hơn — giữ nguyên .env")
        print(f"    Val loss cũ : {old_val_loss:.4f}  ({current_model_path.name})  ← đang dùng")
        print(f"    Val loss mới: {new_val_loss:.4f}  ({new_model_path.name})  ← bỏ qua")
    print("─" * 50)


def _read_best_val_loss(model_path: Path) -> float | None:
    """
    Đọc val_loss của một model từ file metrics.json trong report/ cùng timestamp.
    Trả về None nếu không tìm thấy.
    """
    if not model_path.exists():
        return None  # Model cũ không tồn tại → coi như chưa có

    # Tên model: lstm_YYMMDDHHMM.h5 → timestamp = YYMMDDHHMM
    stem = model_path.stem  # "lstm_2605292154"
    parts = stem.split("_", 1)
    if len(parts) < 2:
        return None
    ts = parts[1]  # "2605292154"

    # Tìm training_log trong report/<ts>/
    report_dir = model_path.parent / "report" / ts
    training_log = report_dir / f"training_log_{ts}.json"

    if training_log.exists():
        try:
            data = json.loads(training_log.read_text(encoding="utf-8"))
            val_losses = data.get("val_loss", [])
            if val_losses:
                return float(min(val_losses))
        except Exception:
            pass

    return None


def _write_model_path_to_env(
    lines: list[str],
    line_idx: int,
    new_model_path: Path,
    project_root: Path,
    env_path: Path,
) -> None:
    """Ghi dòng MODEL_PATH mới vào .env, giữ nguyên toàn bộ dòng khác."""
    # Lưu relative path từ project_root vào .env (giống format cũ)
    try:
        rel_path = new_model_path.relative_to(project_root)
    except ValueError:
        rel_path = new_model_path  # fallback: absolute

    lines[line_idx] = f"MODEL_PATH={rel_path.as_posix()}"
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(".env đã cập nhật: MODEL_PATH=%s", rel_path.as_posix())
    print(f"  .env cập nhật  : MODEL_PATH={rel_path.as_posix()}")
    print(f"  → Chạy 'docker-compose restart backend' để áp dụng")


if __name__ == "__main__":
    history = train()

    # In tóm tắt cuối
    n_epochs = len(history["val_loss"])
    new_val_loss = min(history["val_loss"])

    print("\n" + "=" * 50)
    print(f"  Epochs thực tế chạy : {n_epochs}")
    print(f"  Val loss tốt nhất   : {new_val_loss:.4f}")
    print(f"  Val accuracy tốt nhất: {max(history['val_accuracy'])*100:.1f}%")
    print(f"  Model: {MODEL_PATH}")
    print("=" * 50)

    # So sánh với model cũ và cập nhật .env nếu tốt hơn
    update_env_if_better(MODEL_PATH, new_val_loss)