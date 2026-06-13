"""
inference.py — Production Inference
=====================================
File này được import bởi backend/app/services/ml_service.py.
KHÔNG import train.py hay evaluate.py — chỉ load model và predict.

Dùng:
    from inference import InferenceEngine

    engine = InferenceEngine()                      # load model 1 lần khi startup
    result = engine.predict(raw_adc_window)         # gọi mỗi khi có data từ BLE
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional
import threading

import sys
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import process_realtime_window, WINDOW_SIZE
from model import load_trained_model, CLASS_NAMES, N_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_PATH        = Path("ml/models/lstm_2606010220.h5")

# Ngưỡng confidence mặc định — window có max_proba < threshold
# sẽ trả về label "uncertain" thay vì predict sai
DEFAULT_CONFIDENCE_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Kết quả predict — trả về dạng dict sạch cho backend dùng
# ---------------------------------------------------------------------------

class PredictionResult:
    """
    Đóng gói kết quả predict thành object có thể serialize sang JSON.
    """
    __slots__ = (
        "label", "label_index", "confidence",
        "probabilities", "is_uncertain", "window_id",
    )

    def __init__(
        self,
        label: str,
        label_index: int,
        confidence: float,
        probabilities: dict,
        is_uncertain: bool,
        window_id: Optional[int] = None,
    ):
        self.label        = label           # "Normal" / "Parkinson" / "Abnormal" / "Uncertain"
        self.label_index  = label_index     # 0 / 1 / 2 / -1
        self.confidence   = confidence      # 0.0 – 1.0
        self.probabilities = probabilities  # {"Normal": 0.8, "Parkinson": 0.15, "Abnormal": 0.05}
        self.is_uncertain = is_uncertain    # True nếu confidence < threshold
        self.window_id    = window_id       # số thứ tự window trong session (tuỳ chọn)

    def to_dict(self) -> dict:
        return {
            "label":         self.label,
            "label_index":   self.label_index,
            "confidence":    round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "is_uncertain":  self.is_uncertain,
            "window_id":     self.window_id,
        }

    def __repr__(self) -> str:
        return (
            f"PredictionResult(label={self.label!r}, "
            f"confidence={self.confidence:.3f}, "
            f"uncertain={self.is_uncertain})"
        )


# ---------------------------------------------------------------------------
# InferenceEngine — singleton pattern, thread-safe
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Load model một lần duy nhất khi khởi động backend.
    Thread-safe: dùng Lock để tránh race condition khi nhiều request đến cùng lúc.

    Cách dùng trong ml_service.py:
        engine = InferenceEngine()          # gọi 1 lần ở module level
        result = engine.predict(raw_window) # gọi mỗi request
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_path: Path = MODEL_PATH, **kwargs):
        """Singleton: chỉ tạo 1 instance dù gọi InferenceEngine() nhiều lần."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        if self._initialized:
            return  # singleton đã init rồi, bỏ qua

        self.model_path           = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self._model               = None
        self._predict_lock        = threading.Lock()
        self._model_metrics       = self._load_metrics()

        self._load_model()
        self._initialized = True

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        log.info("InferenceEngine: loading model từ %s", self.model_path)
        self._model = load_trained_model(self.model_path)
        # Warm-up: chạy 1 dummy prediction để TF compile graph trước
        dummy = np.zeros((1, WINDOW_SIZE, 4), dtype=np.float32)
        self._model.predict(dummy, verbose=0)
        log.info("InferenceEngine: ready — threshold=%.2f", self.confidence_threshold)

    def _load_metrics(self) -> Optional[dict]:
        """Load metrics.json từ evaluate.py để backend có thể expose model info."""
        import glob
        pattern = str(Path("ml/models/report") / "*" / "evaluation" / "metrics.json")
        candidates = sorted(glob.glob(pattern), reverse=True)
        if candidates:
            with open(candidates[0], encoding="utf-8") as f:
                return json.load(f)
        fallback = Path("ml/models/evaluation/metrics.json")
        if fallback.exists():
            with open(fallback, encoding="utf-8") as f:
                return json.load(f)
        log.warning("Không tìm thấy metrics.json — model chưa được evaluate")
        return None

    # ------------------------------------------------------------------
    # Predict — 1 window
    # ------------------------------------------------------------------

    def predict(
        self,
        raw_adc: np.ndarray,
        window_id: Optional[int] = None,
    ) -> PredictionResult:
        """
        Nhận data thô từ ESP32/ADS1115, trả về PredictionResult.

        Tham số:
            raw_adc   : np.ndarray shape (100, 4), dtype float32
                        Col 0,1 = ADS1115 (0–32767)
                        Col 2,3 = ESP32 ADC (0–4095)
            window_id : số thứ tự window trong session, để frontend track

        Quy trình nội bộ:
            raw_adc (100,4)
              → process_realtime_window()   # normalize + filter
              → model.predict()             # (1,3) softmax
              → so sánh với threshold
              → PredictionResult
        """
        # Preprocessing — dùng đúng hàm trong preprocess.py
        X = process_realtime_window(raw_adc)    # (1, 100, 4)

        # Inference — thread-safe
        with self._predict_lock:
            proba = self._model.predict(X, verbose=0)   # (1, 3)

        proba = proba[0]    # (3,) — bỏ batch dimension
        best_idx    = int(np.argmax(proba))
        confidence  = float(proba[best_idx])
        is_uncertain = confidence < self.confidence_threshold

        probabilities = {CLASS_NAMES[i]: float(proba[i]) for i in range(N_CLASSES)}

        if is_uncertain:
            label       = "Uncertain"
            label_index = -1
        else:
            label       = CLASS_NAMES[best_idx]
            label_index = best_idx

        return PredictionResult(
            label        = label,
            label_index  = label_index,
            confidence   = confidence,
            probabilities = probabilities,
            is_uncertain  = is_uncertain,
            window_id    = window_id,
        )

    # ------------------------------------------------------------------
    # Predict batch — nhiều window cùng lúc (dùng cho cuối session)
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        raw_adc_batch: np.ndarray,
    ) -> list[PredictionResult]:
        """
        Predict nhiều window cùng lúc — hiệu quả hơn gọi predict() nhiều lần.
        Dùng ở cuối session khi backend xử lý toàn bộ data một lần.

        Tham số:
            raw_adc_batch : shape (N, 100, 4)

        Trả về:
            list[PredictionResult] độ dài N
        """
        if raw_adc_batch.ndim != 3 or raw_adc_batch.shape[1:] != (WINDOW_SIZE, 4):
            raise ValueError(
                f"Kích thước không đúng: nhận {raw_adc_batch.shape}, "
                f"cần (N, {WINDOW_SIZE}, 4)"
            )

        # Preprocessing từng window
        X_list = [
            process_realtime_window(raw_adc_batch[i])[0]   # bỏ batch dim → (100,4)
            for i in range(len(raw_adc_batch))
        ]
        X = np.stack(X_list, axis=0)    # (N, 100, 4)

        with self._predict_lock:
            probas = self._model.predict(X, batch_size=32, verbose=0)   # (N, 3)

        results = []
        for i, proba in enumerate(probas):
            best_idx     = int(np.argmax(proba))
            confidence   = float(proba[best_idx])
            is_uncertain = confidence < self.confidence_threshold
            probabilities = {CLASS_NAMES[j]: float(proba[j]) for j in range(N_CLASSES)}

            results.append(PredictionResult(
                label        = "Uncertain" if is_uncertain else CLASS_NAMES[best_idx],
                label_index  = -1 if is_uncertain else best_idx,
                confidence   = confidence,
                probabilities = probabilities,
                is_uncertain  = is_uncertain,
                window_id    = i,
            ))

        return results

    # ------------------------------------------------------------------
    # Aggregate kết quả toàn session → 1 diagnosis
    # ------------------------------------------------------------------

    def aggregate_session(
        self,
        results: list[PredictionResult],
        min_confident_ratio: float = 0.5,
    ) -> dict:
        """
        Sau khi predict xong toàn bộ window của 1 session (5–10 phút đi bộ),
        tổng hợp thành 1 kết quả chẩn đoán cuối.

        Chiến lược: majority vote trên các window confident (không uncertain).
        Nếu < min_confident_ratio window đủ confident → kết quả "Insufficient data".

        Trả về dict:
            {
                "diagnosis":         "Parkinson",
                "confidence_mean":   0.847,
                "label_index":       1,
                "vote_distribution": {"Normal": 12, "Parkinson": 45, "Abnormal": 3},
                "total_windows":     70,
                "confident_windows": 60,
                "confident_ratio":   0.857,
                "sufficient_data":   True,
            }
        """
        total = len(results)
        if total == 0:
            return {"diagnosis": "No data", "sufficient_data": False}

        confident = [r for r in results if not r.is_uncertain]
        confident_ratio = len(confident) / total

        if confident_ratio < min_confident_ratio:
            log.warning(
                "Chỉ có %.1f%% window confident — không đủ để chẩn đoán",
                confident_ratio * 100,
            )
            return {
                "diagnosis":       "Insufficient data",
                "confident_ratio": round(confident_ratio, 4),
                "total_windows":   total,
                "sufficient_data": False,
            }

        # Majority vote
        vote_counts = {name: 0 for name in CLASS_NAMES}
        for r in confident:
            vote_counts[r.label] += 1

        diagnosis    = max(vote_counts, key=vote_counts.get)
        diagnosis_idx = CLASS_NAMES.index(diagnosis)
        conf_mean    = float(np.mean([r.confidence for r in confident]))

        return {
            "diagnosis":         diagnosis,
            "confidence_mean":   round(conf_mean, 4),
            "label_index":       diagnosis_idx,
            "vote_distribution": vote_counts,
            "total_windows":     total,
            "confident_windows": len(confident),
            "confident_ratio":   round(confident_ratio, 4),
            "sufficient_data":   True,
        }

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def model_info(self) -> dict:
        """Trả về thông tin model để backend expose qua GET /model/info."""
        info = {
            "model_path":            str(self.model_path),
            "confidence_threshold":  self.confidence_threshold,
            "classes":               CLASS_NAMES,
            "input_shape":           [WINDOW_SIZE, 4],
            "n_parameters":          int(self._model.count_params()) if self._model else 0,
        }
        if self._model_metrics:
            info["eval_accuracy"]    = self._model_metrics.get("accuracy")
            info["eval_roc_auc"]     = self._model_metrics.get("roc_auc_macro")
            info["eval_macro_f1"]    = self._model_metrics.get("macro_f1")
        return info


# ---------------------------------------------------------------------------
# Chạy trực tiếp — kiểm tra nhanh không cần model thật
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    log.info("=== Kiểm tra inference.py (không cần model thật) ===")

    # Test PredictionResult
    r = PredictionResult(
        label="Parkinson",
        label_index=1,
        confidence=0.87,
        probabilities={"Normal": 0.08, "Parkinson": 0.87, "Abnormal": 0.05},
        is_uncertain=False,
        window_id=42,
    )
    d = r.to_dict()
    assert d["label"] == "Parkinson"
    assert d["window_id"] == 42
    assert abs(sum(d["probabilities"].values()) - 1.0) < 0.01
    log.info("PredictionResult.to_dict(): OK — %s", d)

    # Test aggregate_session với mock results
    mock_results = (
        [PredictionResult("Parkinson", 1, 0.85, {}, False)] * 45 +
        [PredictionResult("Normal",    0, 0.80, {}, False)] * 12 +
        [PredictionResult("Abnormal",  2, 0.75, {}, False)] *  3 +
        [PredictionResult("Uncertain", -1, 0.55, {}, True)] * 10
    )

    # Tạo engine giả (không load model thật)
    engine = object.__new__(InferenceEngine)
    engine.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    engine._model = None
    engine._initialized = True
    engine._model_metrics = None

    agg = engine.aggregate_session(mock_results)
    assert agg["diagnosis"] == "Parkinson", f"Sai: {agg['diagnosis']}"
    assert agg["sufficient_data"] is True
    assert agg["total_windows"] == 70
    log.info("aggregate_session(): OK — %s", agg)

    # Test insufficient data
    few_results = [PredictionResult("Uncertain", -1, 0.40, {}, True)] * 10
    agg2 = engine.aggregate_session(few_results)
    assert agg2["sufficient_data"] is False
    log.info("aggregate_session (insufficient): OK — %s", agg2)

    log.info("=== Tất cả test PASSED ===")