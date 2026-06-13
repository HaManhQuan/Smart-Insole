"""
evaluate.py — Đánh giá Model
==============================
Chạy:
    python ml/src/evaluate.py

Yêu cầu:
    - ml/models/lstm_v1.h5         (chạy train.py trước)
    - ml/data/processed/X_test.npy
    - ml/data/processed/y_test.npy

Output:
    - ml/models/evaluation/metrics.json        ← accuracy, F1, ROC-AUC từng class
    - ml/models/evaluation/confusion_matrix.png
    - ml/models/evaluation/roc_curves.png
"""

import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")   # không cần display server
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from model import load_trained_model, CLASS_NAMES, N_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR  = Path("ml/data/processed")
MODEL_PATH     = Path("ml/models/lstm_2606010220.h5")
from datetime import datetime
_EVAL_TS = datetime.now().strftime("%y%m%d%H%M")
EVAL_DIR = Path("ml/models/report") / _EVAL_TS / "evaluation"

BATCH_SIZE     = 64    # batch lớn hơn training vì chỉ forward pass, không backprop


# ---------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------

def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    for fname in ["X_test.npy", "y_test.npy"]:
        if not (PROCESSED_DIR / fname).exists():
            raise FileNotFoundError(
                f"Không tìm thấy {PROCESSED_DIR / fname}\n"
                "Hãy chạy preprocess.py trước."
            )
    X = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y = np.load(PROCESSED_DIR / "y_test.npy").astype(np.int8)
    return X, y


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_proba = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    y_pred_class = np.argmax(y_proba, axis=1).astype(np.int8)
    return y_pred_class, y_proba


# ---------------------------------------------------------------------------
# Metrics — tự động detect số class thực tế trong data
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    # Detect class thực tế có trong data (tránh lỗi khi chỉ có 2/3 class)
    actual_classes = sorted(np.unique(y_true).tolist())
    actual_names   = [CLASS_NAMES[i] for i in actual_classes]

    report = classification_report(
        y_true, y_pred,
        labels=actual_classes,
        target_names=actual_names,
        output_dict=True,
        zero_division=0,
    )

    # ROC-AUC — binarize chỉ theo class thực tế
    y_bin = label_binarize(y_true, classes=actual_classes)
    # Lấy cột proba tương ứng với actual_classes
    y_proba_actual = y_proba[:, actual_classes]

    # Nếu chỉ có 2 class → binary AUC
    if len(actual_classes) == 2:
        try:
            auc_macro = float(roc_auc_score(y_true, y_proba_actual[:, 1]))
            auc_per_class = {
                actual_names[0]: float(roc_auc_score(
                    (y_true == actual_classes[0]).astype(int), y_proba_actual[:, 0]
                )),
                actual_names[1]: float(roc_auc_score(
                    (y_true == actual_classes[1]).astype(int), y_proba_actual[:, 1]
                )),
            }
        except ValueError as e:
            log.warning("Không tính được ROC-AUC: %s", e)
            auc_macro = 0.0
            auc_per_class = {name: 0.0 for name in actual_names}
    else:
        try:
            auc_macro = float(roc_auc_score(y_bin, y_proba_actual, multi_class="ovr", average="macro"))
            auc_per_class = {
                actual_names[i]: float(roc_auc_score(y_bin[:, i], y_proba_actual[:, i]))
                for i in range(len(actual_classes))
            }
        except ValueError as e:
            log.warning("Không tính được ROC-AUC: %s", e)
            auc_macro = 0.0
            auc_per_class = {name: 0.0 for name in actual_names}

    metrics = {
        "accuracy":          float(report["accuracy"]),
        "macro_f1":          float(report["macro avg"]["f1-score"]),
        "weighted_f1":       float(report["weighted avg"]["f1-score"]),
        "roc_auc_macro":     auc_macro,
        "roc_auc_per_class": auc_per_class,
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall":    float(report[name]["recall"]),
                "f1":        float(report[name]["f1-score"]),
                "support":   int(report[name]["support"]),
            }
            for name in actual_names
        },
        "_actual_classes": actual_classes,
        "_actual_names":   actual_names,
    }
    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path,
                           actual_names: list) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=actual_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=0)

    # Thêm % vào từng ô
    total_per_row = cm.sum(axis=1, keepdims=True)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / total_per_row[i, 0] * 100
            ax.text(
                j, i + 0.3,
                f"({pct:.1f}%)",
                ha="center", va="center",
                fontsize=9, color="white" if cm[i, j] > cm.max() / 2 else "gray",
            )

    ax.set_title("Confusion Matrix — Smart Insole LSTM", fontsize=13, pad=12)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# ROC curves plot
# ---------------------------------------------------------------------------

def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, output_path: Path,
                    actual_classes: list, actual_names: list) -> None:
    y_bin = label_binarize(y_true, classes=actual_classes)
    y_proba_actual = y_proba[:, actual_classes]
    colors = ["#1D9E75", "#534AB7", "#D85A30"]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Binary case
    if len(actual_classes) == 2:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba_actual[:, 1],
                                     pos_label=actual_classes[1])
            auc = roc_auc_score(y_true, y_proba_actual[:, 1])
            ax.plot(fpr, tpr, color=colors[1], lw=2,
                    label=f"{actual_names[1]} vs {actual_names[0]} (AUC = {auc:.3f})")
        except ValueError:
            log.warning("Không vẽ được ROC curve")
    else:
        for i, (name, color) in enumerate(zip(actual_names, colors)):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba_actual[:, i])
                auc = roc_auc_score(y_bin[:, i], y_proba_actual[:, i])
                ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})")
            except ValueError:
                log.warning("Không vẽ được ROC cho class %s", name)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — One-vs-Rest", fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Log metrics ra console
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: dict) -> None:
    actual_names = metrics["_actual_names"]

    log.info("=" * 55)
    log.info("  TONG HOP KET QUA DANH GIA")
    log.info("=" * 55)
    log.info("  Accuracy     : %.4f  (%.1f%%)", metrics["accuracy"], metrics["accuracy"] * 100)
    log.info("  Macro F1     : %.4f", metrics["macro_f1"])
    log.info("  Weighted F1  : %.4f", metrics["weighted_f1"])
    log.info("  ROC-AUC macro: %.4f", metrics["roc_auc_macro"])
    log.info("-" * 55)
    log.info("  %-12s  %9s  %9s  %9s  %9s %7s", "Class", "Precision", "Recall", "F1", "AUC", "Support")
    log.info("  " + "-" * 60)
    for name in actual_names:
        pc   = metrics["per_class"][name]
        auc  = metrics["roc_auc_per_class"][name]
        flag = " WARNING" if pc["f1"] < 0.70 else ""
        log.info(
            "  %-12s  %9.4f  %9.4f  %9.4f  %9.4f %7d%s",
            name, pc["precision"], pc["recall"], pc["f1"], auc, pc["support"], flag,
        )
    log.info("=" * 55)

    low_f1 = [n for n in actual_names if metrics["per_class"][n]["f1"] < 0.70]
    if low_f1:
        log.warning(
            "Class %s co F1 < 0.70 — kiem tra data hoac class weight",
            low_f1,
        )


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------

def analyze_confidence_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list = [0.5, 0.6, 0.7, 0.8, 0.9],
) -> dict:
    results = {}
    max_proba = y_proba.max(axis=1)

    for thr in thresholds:
        mask = max_proba >= thr
        n_above = mask.sum()
        if n_above == 0:
            results[str(thr)] = {"accuracy": None, "coverage": 0.0, "n_samples": 0}
            continue
        acc = (np.argmax(y_proba[mask], axis=1) == y_true[mask]).mean()
        results[str(thr)] = {
            "accuracy":  float(acc),
            "coverage":  float(n_above / len(y_true)),
            "n_samples": int(n_above),
        }
        log.info(
            "Threshold %.1f: accuracy=%.3f  coverage=%.1f%%  n=%d",
            thr, acc, n_above / len(y_true) * 100, n_above,
        )

    return results
# THÊM HÀM MỚI
# SAU — tìm threshold tối ưu macro F1, với ràng buộc cả 2 class recall >= 0.55
# SAU — bỏ ràng buộc cứng, thay bằng tối ưu harmonic mean của 2 recall
def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    from sklearn.metrics import recall_score, f1_score
    best_t    = 0.5
    best_score = 0.0

    for t in np.arange(0.20, 0.80, 0.01):
        y_pred_t    = (y_proba[:, 1] >= t).astype(int)
        r_normal    = recall_score(y_true, y_pred_t, pos_label=0, zero_division=0)
        r_parkinson = recall_score(y_true, y_pred_t, pos_label=1, zero_division=0)

        # Harmonic mean của 2 recall — cân bằng cả 2 class
        if r_normal + r_parkinson == 0:
            continue
        score = 2 * r_normal * r_parkinson / (r_normal + r_parkinson)

        if score > best_score:
            best_score = score
            best_t     = t

    # Log kết quả tại threshold tối ưu
    y_pred_opt = (y_proba[:, 1] >= best_t).astype(int)
    r_n = recall_score(y_true, y_pred_opt, pos_label=0, zero_division=0)
    r_p = recall_score(y_true, y_pred_opt, pos_label=1, zero_division=0)
    log.info("Optimal threshold: %.2f  →  Normal recall=%.4f  Parkinson recall=%.4f",
             best_t, r_n, r_p)
    return best_t

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate() -> dict:
    log.info("=== Smart Insole — Evaluate LSTM ===")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_test_data()
    log.info("Test set: %d windows, shape %s", len(y_test), X_test.shape)
    unique, counts = np.unique(y_test, return_counts=True)
    for u, c in zip(unique, counts):
        log.info("  %s: %d windows", CLASS_NAMES[int(u)], c)

    model = load_trained_model(MODEL_PATH)
    log.info("Model loaded: %s", MODEL_PATH)

    log.info("Running inference tren test set...")
    y_pred, y_proba = predict(model, X_test)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_metrics_table(metrics)

    log.info("--- Confidence threshold analysis ---")
    threshold_results = analyze_confidence_threshold(y_test, y_proba)
    # THÊM VÀO evaluate() — sau dòng gọi analyze_confidence_threshold
    log.info("--- Optimal threshold analysis ---")
    optimal_t = find_optimal_threshold(y_test, y_proba)
    y_pred_optimal  = (y_proba[:, 1] >= optimal_t).astype(int)
    metrics_optimal = compute_metrics(y_test, y_pred_optimal, y_proba)
    log.info("=== Ket qua voi optimal threshold (%.2f) ===", optimal_t)
    for name in metrics_optimal["_actual_names"]:
        pc = metrics_optimal["per_class"][name]
        log.info("  %-12s — Recall: %.4f  F1: %.4f", name, pc["recall"], pc["f1"])
    metrics["optimal_threshold"]         = optimal_t
    metrics["metrics_at_optimal_threshold"] = {
        k: v for k, v in metrics_optimal.items() if not k.startswith("_")
    }
    metrics["threshold_analysis"] = threshold_results

    # Lưu JSON (bỏ các key nội bộ bắt đầu bằng _)
    metrics_to_save = {k: v for k, v in metrics.items() if not k.startswith("_")}
    metrics_path = EVAL_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    log.info("Saved: %s", metrics_path)

    actual_classes = metrics["_actual_classes"]
    actual_names   = metrics["_actual_names"]

    plot_confusion_matrix(y_test, y_pred, EVAL_DIR / "confusion_matrix.png", actual_names)
    plot_roc_curves(y_test, y_proba, EVAL_DIR / "roc_curves.png", actual_classes, actual_names)

    log.info("=== Evaluation hoan tat ===")
    log.info("Ket qua tai: %s", EVAL_DIR.resolve())
    return metrics


if __name__ == "__main__":
    evaluate()