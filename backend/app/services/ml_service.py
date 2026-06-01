"""
ml_service.py — ML Service Bridge
====================================
Cầu nối giữa inference.py và FastAPI routers.
Routers KHÔNG import inference.py trực tiếp — chỉ import từ file này.

Lý do tách biệt:
  - ml_service.py xử lý business logic (validate, log, error handling)
  - inference.py chỉ thuần ML (load model, predict)
  - Dễ mock khi test routers mà không cần load TensorFlow

Import trong routers:
    from app.services.ml_service import ml_service
    result = await ml_service.predict_window(window)
    diagnosis = ml_service.finalize_session(session_id, predictions)
"""

from __future__ import annotations

import sys
import logging
import time
from pathlib import Path
from typing import Optional
import numpy as np

_ML_SRC = Path(__file__).parents[3] / "ml" / "src"
if str(_ML_SRC) not in sys.path:
    sys.path.insert(0, str(_ML_SRC))

from app.config import settings
from app.models.schemas import (
    SensorWindow,
    WindowPrediction,
    SessionDiagnosis,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import TensorFlow — chỉ load khi cần, không lúc import module
# ---------------------------------------------------------------------------

_engine = None


def _get_engine():
    """
    Lazy-load InferenceEngine.
    Tách ra để:
      1. Backend start nhanh hơn (không chờ TF load lúc import)
      2. Dễ mock trong unit test
      3. Nếu model chưa có, chỉ lỗi khi predict — không crash toàn bộ app
    """
    global _engine
    if _engine is None:
        try:
            from inference import InferenceEngine
            _engine = InferenceEngine(
                model_path=settings.MODEL_PATH,
                confidence_threshold=settings.CONFIDENCE_THRESHOLD,
            )
            log.info("InferenceEngine loaded: %s", settings.MODEL_PATH)
        except FileNotFoundError as e:
            log.error("Model không tìm thấy: %s", e)
            raise
        except Exception as e:
            log.error("Lỗi khi load InferenceEngine: %s", e)
            raise
    return _engine


# ---------------------------------------------------------------------------
# MLService
# ---------------------------------------------------------------------------

class MLService:
    """
    Service layer cho ML operations.
    Instantiate 1 lần ở module level → dùng như singleton.

    Nhiệm vụ:
      - Validate input trước khi predict
      - Gọi InferenceEngine
      - Đổi PredictionResult → Pydantic schema (WindowPrediction)
      - Log latency và kết quả
      - Aggregate kết quả session → SessionDiagnosis
    """

    # ------------------------------------------------------------------
    # Predict 1 window
    # ------------------------------------------------------------------

    async def predict_window(
        self,
        window: SensorWindow,
    ) -> tuple[WindowPrediction, float]:
        """
        Nhận SensorWindow từ router, trả về (WindowPrediction, latency_ms).

        Quy trình:
            SensorWindow.to_numpy() → (100, 4) raw ADC
            → InferenceEngine.predict()
            → PredictionResult
            → WindowPrediction (Pydantic schema)

        Raises:
            RuntimeError : model chưa load được
            ValueError   : window data có vấn đề
        """
        engine = _get_engine()

        raw_adc = window.to_numpy()   # (100, 4) float32

        t0 = time.perf_counter()
        result = engine.predict(raw_adc, window_id=window.window_id)
        latency_ms = (time.perf_counter() - t0) * 1000

        prediction = WindowPrediction(
            window_id     = window.window_id,
            label         = result.label,
            label_index   = result.label_index,
            confidence    = result.confidence,
            is_uncertain  = result.is_uncertain,
            probabilities = result.probabilities,
        )

        log.info(
            "Predict session=%d window=%d → %s (conf=%.3f, uncertain=%s, %.1fms)",
            window.session_id,
            window.window_id,
            result.label,
            result.confidence,
            result.is_uncertain,
            latency_ms,
        )

        return prediction, latency_ms

    # ------------------------------------------------------------------
    # Predict batch (cuối session)
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        windows: list[SensorWindow],
    ) -> list[WindowPrediction]:
        """
        Predict nhiều window một lúc — hiệu quả hơn gọi predict_window() nhiều lần.
        Dùng khi client gửi toàn bộ data sau khi bệnh nhân đi xong.
        """
        if not windows:
            return []

        engine = _get_engine()

        # Stack tất cả window thành (N, 100, 4)
        raw_batch = np.stack(
            [w.to_numpy() for w in windows],
            axis=0,
        )

        t0 = time.perf_counter()
        results = engine.predict_batch(raw_batch)
        latency_ms = (time.perf_counter() - t0) * 1000

        log.info(
            "Batch predict: %d windows → %.1fms (%.2fms/window)",
            len(windows), latency_ms, latency_ms / len(windows),
        )

        predictions = [
            WindowPrediction(
                window_id     = r.window_id if r.window_id is not None else i,
                label         = r.label,
                label_index   = r.label_index,
                confidence    = r.confidence,
                is_uncertain  = r.is_uncertain,
                probabilities = r.probabilities,
            )
            for i, r in enumerate(results)
        ]

        return predictions

    # ------------------------------------------------------------------
    # Aggregate session → diagnosis
    # ------------------------------------------------------------------

    def finalize_session(
        self,
        session_id: int,
        predictions: list[WindowPrediction],
    ) -> SessionDiagnosis:
        """
        Tổng hợp tất cả WindowPrediction của 1 session thành 1 chẩn đoán.

        Gọi sau khi bệnh nhân đi xong (session ended).
        Kết quả này được lưu vào PostgreSQL và hiển thị cho bác sĩ.

        Chiến lược: majority vote trên các window không uncertain.
        """
        engine = _get_engine()

        # Chuyển WindowPrediction → PredictionResult để dùng aggregate_session()
        from inference import PredictionResult

        results = [
            PredictionResult(
                label         = p.label,
                label_index   = p.label_index,
                confidence    = p.confidence,
                probabilities = p.probabilities,
                is_uncertain  = p.is_uncertain,
                window_id     = p.window_id,
            )
            for p in predictions
        ]

        agg = engine.aggregate_session(
            results,
            min_confident_ratio=settings.MIN_CONFIDENT_RATIO,
        )

        log.info(
            "Session %d finalized → %s (conf=%.3f, %d/%d windows confident)",
            session_id,
            agg.get("diagnosis"),
            agg.get("confidence_mean", 0),
            agg.get("confident_windows", 0),
            agg.get("total_windows", 0),
        )

        # Map dict từ aggregate_session() → Pydantic SessionDiagnosis
        if not agg.get("sufficient_data", False):
            return SessionDiagnosis(
                diagnosis        = agg.get("diagnosis", "Insufficient data"),
                label_index      = -1,
                confidence_mean  = 0.0,
                vote_distribution = {},
                total_windows    = agg.get("total_windows", len(predictions)),
                confident_windows = 0,
                confident_ratio  = agg.get("confident_ratio", 0.0),
                sufficient_data  = False,
            )

        return SessionDiagnosis(
            diagnosis        = agg["diagnosis"],
            label_index      = agg["label_index"],
            confidence_mean  = agg["confidence_mean"],
            vote_distribution = agg["vote_distribution"],
            total_windows    = agg["total_windows"],
            confident_windows = agg["confident_windows"],
            confident_ratio  = agg["confident_ratio"],
            sufficient_data  = True,
        )

    # ------------------------------------------------------------------
    # Tích lũy sample từ WebSocket → window khi đủ 100 samples
    # ------------------------------------------------------------------

    def create_window_accumulator(self, session_id: int) -> "WindowAccumulator":
        """
        Factory method tạo accumulator cho 1 session WebSocket.
        Router WebSocket gọi hàm này khi connection mở.

        Dùng:
            acc = ml_service.create_window_accumulator(session_id)
            # mỗi sample BLE đến:
            window = acc.add_sample(s1, s2, s3, s4, timestamp)
            if window is not None:
                prediction, _ = ml_service.predict_window(window)
        """
        return WindowAccumulator(
            session_id  = session_id,
            buffer_size = settings.WS_BUFFER_SIZE,
        )

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    @property
    def model_info(self) -> dict:
        """Trả về thông tin model — dùng cho GET /model/info."""
        try:
            return _get_engine().model_info
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}

    def is_ready(self) -> bool:
        """Kiểm tra model có sẵn sàng predict không — dùng cho GET /health."""
        try:
            _get_engine()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# WindowAccumulator — tích lũy sample BLE thành window
# ---------------------------------------------------------------------------

class WindowAccumulator:
    """
    Nhận từng ADC sample (50Hz) từ WebSocket, tích lũy thành window 100 samples.
    Mỗi khi đủ buffer_size samples → yield SensorWindow → reset (sliding).

    Overlap: giữ lại 50 sample cuối sau mỗi window (50% overlap, stride=50).
    Khớp với cách train: STRIDE=50 trong preprocess.py.
    """

    def __init__(self, session_id: int, buffer_size: int = 100):
        self.session_id  = session_id
        self.buffer_size = buffer_size
        self.stride      = buffer_size // 2   # 50% overlap

        self._s1: list[float] = []
        self._s2: list[float] = []
        self._s3: list[float] = []
        self._s4: list[float] = []
        self._timestamps: list = []
        self._window_count = 0

    def add_sample(
        self,
        s1: float, s2: float, s3: float, s4: float,
        timestamp,
    ) -> Optional[SensorWindow]:
        """
        Thêm 1 sample. Trả về SensorWindow khi đủ buffer_size samples,
        None nếu chưa đủ.
        """
        self._s1.append(s1)
        self._s2.append(s2)
        self._s3.append(s3)
        self._s4.append(s4)
        self._timestamps.append(timestamp)

        if len(self._s1) >= self.buffer_size:
            window = SensorWindow(
                session_id = self.session_id,
                window_id  = self._window_count,
                timestamp  = self._timestamps[-1],
                s1         = list(self._s1[-self.buffer_size:]),
                s2         = list(self._s2[-self.buffer_size:]),
                s3         = list(self._s3[-self.buffer_size:]),
                s4         = list(self._s4[-self.buffer_size:]),
            )
            self._window_count += 1

            # Giữ lại stride cuối (50 sample) — sliding window overlap
            self._s1         = self._s1[-self.stride:]
            self._s2         = self._s2[-self.stride:]
            self._s3         = self._s3[-self.stride:]
            self._s4         = self._s4[-self.stride:]
            self._timestamps = self._timestamps[-self.stride:]

            return window

        return None

    def flush(self) -> Optional[SensorWindow]:
        """
        Tạo window từ những sample còn lại khi session kết thúc.
        Padding bằng 0 nếu chưa đủ buffer_size.
        Trả về None nếu buffer trống.
        """
        if len(self._s1) == 0:
            return None

        # Pad bằng 0 tới buffer_size
        deficit = self.buffer_size - len(self._s1)
        s1 = self._s1 + [0.0] * deficit
        s2 = self._s2 + [0.0] * deficit
        s3 = self._s3 + [0.0] * deficit
        s4 = self._s4 + [0.0] * deficit
        ts = self._timestamps[-1] if self._timestamps else __import__('datetime').datetime.now()

        return SensorWindow(
            session_id = self.session_id,
            window_id  = self._window_count,
            timestamp  = ts,
            s1=s1, s2=s2, s3=s3, s4=s4,
        )

    @property
    def sample_count(self) -> int:
        return len(self._s1)

    @property
    def window_count(self) -> int:
        return self._window_count


# ---------------------------------------------------------------------------
# Singleton — import và dùng trực tiếp trong routers
# ---------------------------------------------------------------------------

ml_service = MLService()