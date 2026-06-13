"""
websocket.py — WebSocket Real-time Streaming
==============================================
WS /api/v1/ws/session/{session_id}

Flow:
  1. Frontend kết nối WebSocket khi bắt đầu đo
  2. Mỗi BLE sample (50Hz) → frontend gửi JSON qua WS
  3. Backend tích lũy đủ 100 samples → predict → gửi kết quả lại
  4. Frontend nhận window_result → cập nhật chart real-time
  5. Frontend gửi session_end → backend flush + tổng hợp diagnosis

Message format:
  Incoming: {"type": "sensor_sample", "payload": {s1, s2, s3, s4, ts}}
           {"type": "session_end",    "payload": {}}
  Outgoing: {"type": "window_result", "payload": WindowPrediction}
            {"type": "diagnosis",     "payload": SessionDiagnosis}
            {"type": "error",         "payload": {"message": "..."}}
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session

from app.models.schemas import (
    SensorReading,
    WSMessageType,
)
from app.services.db_service import (
    get_db,
    get_session_by_id,
    get_influx,
    save_prediction,
    InfluxDBService,
    get_predictions_by_session,
)
from app.services.ml_service import ml_service, WindowAccumulator

log = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


# ---------------------------------------------------------------------------
# Connection Manager — theo dõi active WebSocket connections
# ---------------------------------------------------------------------------

class ConnectionManager:
    """
    Quản lý tất cả WebSocket connections đang mở.
    Đảm bảo không có 2 connection cho cùng 1 session.
    """

    def __init__(self):
        # session_id → WebSocket
        self._active: dict[int, WebSocket] = {}

    async def connect(self, session_id: int, ws: WebSocket) -> bool:
        """
        Chấp nhận kết nối mới. Trả về False nếu session đã có connection.
        """
        if session_id in self._active:
            await ws.close(code=1008, reason=f"Session {session_id} đang có kết nối khác")
            return False
        await ws.accept()
        self._active[session_id] = ws
        log.info("WS connected: session=%d (total=%d)", session_id, len(self._active))
        return True

    def disconnect(self, session_id: int) -> None:
        self._active.pop(session_id, None)
        log.info("WS disconnected: session=%d (total=%d)", session_id, len(self._active))

    @property
    def active_count(self) -> int:
        return len(self._active)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Helper: gửi JSON message
# ---------------------------------------------------------------------------

async def _send(ws: WebSocket, msg_type: str, payload: dict) -> None:
    await ws.send_text(json.dumps({"type": msg_type, "payload": payload}))


async def _send_error(ws: WebSocket, message: str) -> None:
    await _send(ws, WSMessageType.ERROR, {"message": message})


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/session/{session_id}")
async def websocket_session(
    session_id: int,
    ws:         WebSocket,
    db:         Session          = Depends(get_db),
    influx:     InfluxDBService  = Depends(get_influx),
):
    """
    WebSocket handler cho 1 phiên đo.

    Vòng lặp chính:
        receive JSON → parse → add_sample → (nếu đủ window) predict → send result
    """
    # Xác nhận session tồn tại và đang active
    session = get_session_by_id(db, session_id)
    if not session or session.status != "active":
        await ws.accept()
        await _send_error(ws, f"Session {session_id} không hợp lệ hoặc đã kết thúc")
        await ws.close()
        return

    # Kết nối
    connected = await manager.connect(session_id, ws)
    if not connected:
        return

    # Tạo accumulator cho session này
    accumulator: WindowAccumulator = ml_service.create_window_accumulator(session_id)

    # Buffer window để ghi InfluxDB batch (tránh ghi từng sample)
    influx_s1, influx_s2, influx_s3, influx_s4 = [], [], [], []
    last_influx_ts = datetime.now(timezone.utc)
    INFLUX_BATCH = 100   # ghi InfluxDB mỗi 100 samples = mỗi 2 giây

    try:
        while True:
            # Nhận message từ frontend
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await _send_error(ws, "Timeout — không nhận được data trong 30 giây")
                break

            try:
                msg = json.loads(raw)
                msg_type = msg.get("type", "")
                payload  = msg.get("payload", {})
            except (json.JSONDecodeError, AttributeError):
                await _send_error(ws, "Định dạng message không hợp lệ — cần JSON")
                continue

            # ── Session end ──────────────────────────────────────────────
            if msg_type == WSMessageType.SESSION_END:
                # Flush buffer còn lại
                leftover = accumulator.flush()
                if leftover:
                    pred, _ = await ml_service.predict_window(leftover)
                    try:
                        save_prediction(db, session_id, pred)
                    except Exception:
                        pass
                    await _send(ws, WSMessageType.WINDOW_RESULT, pred.model_dump())

                # Tổng hợp chẩn đoán
                pred_rows = get_predictions_by_session(db, session_id)

                from app.models.schemas import WindowPrediction
                predictions = [
                    WindowPrediction(
                        window_id=r.window_id, label=r.label,
                        label_index=r.label_index, confidence=r.confidence,
                        is_uncertain=r.is_uncertain,
                        probabilities={"Normal": r.prob_normal,
                                       "Parkinson": r.prob_parkinson,
                                       "Abnormal": r.prob_abnormal},
                    )
                    for r in pred_rows
                ]
                diagnosis = ml_service.finalize_session(session_id, predictions)
                await _send(ws, WSMessageType.DIAGNOSIS, diagnosis.model_dump())
                break

            # ── Sensor sample ─────────────────────────────────────────────
            elif msg_type == WSMessageType.SENSOR_SAMPLE:
                try:
                    reading = SensorReading(**payload)
                except Exception as e:
                    await _send_error(ws, f"Sensor data không hợp lệ: {e}")
                    continue

                ts = datetime.fromtimestamp(reading.ts / 1000.0, tz=timezone.utc)

                # Tích lũy cho InfluxDB
                influx_s1.append(reading.s1)
                influx_s2.append(reading.s2)
                influx_s3.append(reading.s3)
                influx_s4.append(reading.s4)

                # Ghi InfluxDB theo batch
                if len(influx_s1) >= INFLUX_BATCH:
                    try:
                        influx.write_sensor_window(
                            session_id = session_id,
                            window_id  = accumulator.window_count,
                            timestamp  = ts,
                            s1_values  = influx_s1,
                            s2_values  = influx_s2,
                            s3_values  = influx_s3,
                            s4_values  = influx_s4,
                        )
                    except Exception as e:
                        log.warning("InfluxDB write error: %s", e)
                    influx_s1.clear(); influx_s2.clear()
                    influx_s3.clear(); influx_s4.clear()
                    last_influx_ts = ts

                # Tích lũy cho ML window
                window = accumulator.add_sample(
                    reading.s1, reading.s2, reading.s3, reading.s4, ts
                )

                if window is not None:
                    pred, latency_ms = await ml_service.predict_window(window)
                    try:
                        save_prediction(db, session_id, pred)
                    except Exception as e:
                        log.warning("DB save prediction error: %s", e)

                    await _send(ws, WSMessageType.WINDOW_RESULT, pred.model_dump())

            else:
                await _send_error(ws, f"Loại message không hỗ trợ: '{msg_type}'")

    except WebSocketDisconnect:
        log.info("WS client disconnected: session=%d", session_id)
    except Exception as e:
        log.error("WS unexpected error session=%d: %s", session_id, e)
        try:
            await _send_error(ws, f"Lỗi server: {e}")
        except Exception:
            pass
    finally:
        manager.disconnect(session_id)
        log.info(
            "WS session=%d done — %d windows, %d samples in buffer",
            session_id, accumulator.window_count, accumulator.sample_count,
        )