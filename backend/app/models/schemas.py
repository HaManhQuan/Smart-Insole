"""
schemas.py — Pydantic Schemas
==============================
Định nghĩa tất cả data models cho FastAPI:
  - Request bodies  (dữ liệu client gửi lên)
  - Response models (dữ liệu server trả về)
  - DB row models   (ánh xạ từ PostgreSQL)

Import trong routers:
    from app.models.schemas import Patient, SessionCreate, PredictionResponse, ...

Quy ước đặt tên:
  XxxCreate   → body khi POST (tạo mới)
  XxxUpdate   → body khi PATCH (cập nhật một phần)
  XxxResponse → response trả về client (có id, timestamps)
  Xxx         → internal / DB row model
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GaitLabel(IntEnum):
    NORMAL     = 0
    PARKINSON  = 1
    ABNORMAL   = 2
    UNCERTAIN  = -1   # confidence < threshold

    @classmethod
    def label_name(cls, value: int) -> str:
        mapping = {0: "Normal", 1: "Parkinson", 2: "Abnormal", -1: "Uncertain"}
        return mapping.get(value, "Unknown")


class SessionStatus(str):
    ACTIVE    = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------

class PatientCreate(BaseModel):
    """Body khi POST /patients — tạo bệnh nhân mới."""
    full_name:   str            = Field(..., min_length=2, max_length=120, examples=["Nguyễn Văn A"])
    birth_year:  int            = Field(..., ge=1900, le=2025, examples=[1955])
    gender:      str            = Field(..., pattern="^(male|female|other)$")
    phone:       Optional[str]  = Field(None, max_length=20)
    notes:       Optional[str]  = Field(None, max_length=500)

    @field_validator("full_name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        return v.strip()


class PatientResponse(BaseModel):
    """Response trả về sau khi tạo hoặc GET /patients/{id}."""
    id:          int
    full_name:   str
    birth_year:  int
    gender:      str
    phone:       Optional[str]
    notes:       Optional[str]
    created_at:  datetime

    model_config = {"from_attributes": True}   # cho phép từ SQLAlchemy ORM object


class PatientListResponse(BaseModel):
    """Response cho GET /patients — danh sách có phân trang."""
    total:    int
    page:     int
    per_page: int
    items:    list[PatientResponse]


# ---------------------------------------------------------------------------
# Session (phiên khám)
# ---------------------------------------------------------------------------

class SessionCreate(BaseModel):
    """Body khi POST /sessions — bắt đầu phiên đo mới."""
    patient_id:  int  = Field(..., gt=0)
    notes:       Optional[str] = Field(None, max_length=300)


class SessionEnd(BaseModel):
    """Body khi PATCH /sessions/{id}/end — kết thúc phiên."""
    notes: Optional[str] = Field(None, max_length=300)


class SessionResponse(BaseModel):
    """Response cho session."""
    id:           int
    patient_id:   int
    status:       str
    notes:        Optional[str]
    started_at:   datetime
    ended_at:     Optional[datetime]
    duration_sec: Optional[float]   # tính từ started_at → ended_at

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Sensor Data
# ---------------------------------------------------------------------------

class SensorWindow(BaseModel):
    """
    1 cửa sổ dữ liệu gồm 100 timestep × 4 sensor.
    Gửi từ frontend → POST /sessions/{id}/data hoặc qua WebSocket.

    s1: ụ ngón cái   — ADS1115 (0–32767)
    s2: ngón cái      — ADS1115 (0–32767)
    s3: ụ ngón út    — ESP32 ADC (0–4095)
    s4: gót chân      — ESP32 ADC (0–4095)
    """
    session_id:  int
    window_id:   int                       = Field(..., ge=0)
    timestamp:   datetime
    s1:          list[float]               = Field(..., min_length=100, max_length=100)
    s2:          list[float]               = Field(..., min_length=100, max_length=100)
    s3:          list[float]               = Field(..., min_length=100, max_length=100)
    s4:          list[float]               = Field(..., min_length=100, max_length=100)

    @model_validator(mode="after")
    def check_sensor_ranges(self) -> SensorWindow:
        for name, values, max_val in [
            ("s1", self.s1, 4095),
            ("s2", self.s2, 26400),
            ("s3", self.s3, 4095),
            ("s4", self.s4, 4095),
        ]:
            if any(v < 0 or v > max_val for v in values):
                raise ValueError(
                    f"Sensor {name}: giá trị ngoài range [0, {max_val}]"
                )
        return self

    def to_numpy(self):
        """Chuyển 4 list → numpy array (100, 4) cho InferenceEngine."""
        import numpy as np
        return np.column_stack([self.s1, self.s2, self.s3, self.s4]).astype(np.float32)


class SensorReading(BaseModel):
    """
    1 sample đơn lẻ từ BLE (50Hz).
    Dùng cho WebSocket stream — frontend gửi từng sample,
    backend tích lũy đủ 100 → tạo SensorWindow.
    """
    s1: float = Field(..., ge=0, le=4095)    # ESP32 ADC 12-bit
    s2: float = Field(..., ge=0, le=26400)   # ADS1115 GAIN_ONE
    s3: float = Field(..., ge=0, le=4095)    # ESP32 ADC 12-bit
    s4: float = Field(..., ge=0, le=4095)    # ESP32 ADC 12-bit
    ts: float = Field(..., description="Unix timestamp (ms)")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Body khi POST /predict — gửi 1 window để inference ngay."""
    session_id: int  = Field(..., gt=0)
    window:     SensorWindow


class WindowPrediction(BaseModel):
    """Kết quả predict cho 1 window đơn lẻ."""
    window_id:     int
    label:         str    = Field(..., examples=["Parkinson"])
    label_index:   int    = Field(..., ge=-1, le=2)
    confidence:    float  = Field(..., ge=0.0, le=1.0)
    is_uncertain:  bool
    probabilities: dict[str, float]   # {"Normal": 0.08, "Parkinson": 0.87, "Abnormal": 0.05}


class SessionDiagnosis(BaseModel):
    """
    Kết quả tổng hợp cuối session — từ aggregate_session() của InferenceEngine.
    Trả về khi PATCH /sessions/{id}/end hoặc GET /sessions/{id}/diagnosis.
    """
    diagnosis:         str    = Field(..., examples=["Parkinson"])
    label_index:       int    = Field(..., ge=-1, le=2)
    confidence_mean:   float  = Field(..., ge=0.0, le=1.0)
    vote_distribution: dict[str, int]   # {"Normal": 12, "Parkinson": 45, "Abnormal": 3}
    total_windows:     int
    confident_windows: int
    confident_ratio:   float  = Field(..., ge=0.0, le=1.0)
    sufficient_data:   bool


class PredictionResponse(BaseModel):
    """Response đầy đủ cho POST /predict."""
    session_id:       int
    window_prediction: WindowPrediction
    saved_to_db:      bool   # True nếu đã ghi vào PostgreSQL thành công


class SessionPredictionHistory(BaseModel):
    """Response cho GET /sessions/{id}/predictions — lịch sử tất cả window."""
    session_id:  int
    total:       int
    predictions: list[WindowPrediction]
    diagnosis:   Optional[SessionDiagnosis]   # None nếu session chưa kết thúc


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class ReportCreate(BaseModel):
    """Body khi POST /reports — tạo báo cáo từ session đã hoàn thành."""
    session_id:       int  = Field(..., gt=0)
    doctor_notes:     Optional[str] = Field(None, max_length=1000)
    include_charts:   bool = True


class ReportResponse(BaseModel):
    """Response sau khi tạo report."""
    id:              int
    session_id:      int
    patient_id:      int
    patient_name:    str
    created_at:      datetime
    diagnosis:       str
    confidence_mean: float
    doctor_notes:    Optional[str]
    pdf_url:         Optional[str]   # URL để download PDF

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Model info (GET /model/info)
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    """Thông tin model đang chạy — expose qua API để bác sĩ biết model version."""
    model_path:             str
    confidence_threshold:   float
    classes:                list[str]
    input_shape:            list[int]
    n_parameters:           int
    eval_accuracy:          Optional[float]
    eval_roc_auc:           Optional[float]
    eval_macro_f1:          Optional[float]


# ---------------------------------------------------------------------------
# WebSocket messages
# ---------------------------------------------------------------------------

class WSMessageType(str):
    SENSOR_SAMPLE  = "sensor_sample"    # frontend → backend: 1 ADC sample
    WINDOW_RESULT  = "window_result"    # backend → frontend: kết quả 1 window
    SESSION_END    = "session_end"      # frontend → backend: kết thúc session
    DIAGNOSIS      = "diagnosis"        # backend → frontend: kết quả cuối
    ERROR          = "error"            # backend → frontend: lỗi


class WSIncoming(BaseModel):
    """Message từ frontend → backend qua WebSocket."""
    type:    str
    payload: dict   # SensorReading.model_dump() hoặc {} nếu SESSION_END


class WSOutgoing(BaseModel):
    """Message từ backend → frontend qua WebSocket."""
    type:    str
    payload: dict   # WindowPrediction.model_dump() hoặc SessionDiagnosis.model_dump()


# ---------------------------------------------------------------------------
# Error response chuẩn
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Trả về khi có lỗi — FastAPI exception handler dùng schema này."""
    error:   str
    detail:  Optional[str] = None
    code:    Optional[int] = None


# ---------------------------------------------------------------------------
# Kiểm tra nhanh
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== Kiểm tra schemas ===\n")

    # PatientCreate validation
    p = PatientCreate(
        full_name="  Nguyễn Văn A  ",
        birth_year=1955,
        gender="male",
        phone="0912345678",
    )
    assert p.full_name == "Nguyễn Văn A"   # strip đã chạy
    print("PatientCreate: OK —", p.model_dump())

    # SensorWindow validation + to_numpy
    import numpy as np
    window_data = {
        "session_id": 1,
        "window_id":  0,
        "timestamp":  datetime.now().isoformat(),
        "s1": [float(i * 100) for i in range(100)],    # ADS1115 range
        "s2": [float(i * 200) for i in range(100)],
        "s3": [float(i * 20)  for i in range(100)],    # ESP32 ADC range
        "s4": [float(i * 30)  for i in range(100)],
    }
    w = SensorWindow(**window_data)
    arr = w.to_numpy()
    assert arr.shape == (100, 4)
    assert arr.dtype == np.float32
    print("SensorWindow.to_numpy(): OK — shape", arr.shape)

    # SensorWindow range validation
    try:
        bad = SensorWindow(**{**window_data, "s1": [99999.0] * 100})   # s1 > 32767
        print("ERROR: should have raised ValidationError")
    except Exception as e:
        print("SensorWindow range check: OK — caught ValidationError")

    # SessionDiagnosis serialize
    diag = SessionDiagnosis(
        diagnosis="Parkinson",
        label_index=1,
        confidence_mean=0.847,
        vote_distribution={"Normal": 12, "Parkinson": 45, "Abnormal": 3},
        total_windows=70,
        confident_windows=60,
        confident_ratio=0.857,
        sufficient_data=True,
    )
    print("SessionDiagnosis: OK —", diag.model_dump_json(indent=2)[:80], "...")

    # GaitLabel helper
    assert GaitLabel.label_name(1) == "Parkinson"
    assert GaitLabel.label_name(-1) == "Uncertain"
    print("GaitLabel.label_name(): OK")

    print("\n=== Tất cả test PASSED ===")