"""
sessions.py — Router phiên khám
=================================
POST  /api/v1/sessions                       ← bắt đầu phiên mới
GET   /api/v1/sessions/{id}                  ← chi tiết phiên
PATCH /api/v1/sessions/{id}/end              ← kết thúc phiên + tổng hợp chẩn đoán
POST  /api/v1/sessions/{id}/data             ← ghi batch sensor window vào InfluxDB
GET   /api/v1/sessions/{id}/predictions      ← lịch sử predict của phiên
GET   /api/v1/sessions/{id}/diagnosis        ← chẩn đoán tổng hợp
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.models.schemas import (
    SessionCreate,
    SessionEnd,
    SessionResponse,
    SensorWindow,
    SessionDiagnosis,
    SessionPredictionHistory,
    WindowPrediction,
    ErrorResponse,
)
from app.services.db_service import (
    get_db,
    get_influx,
    create_session,
    get_session_by_id,
    end_session,
    get_patient,
    get_predictions_by_session,
    save_report,
    InfluxDBService,
)
from app.services.ml_service import ml_service

router = APIRouter(prefix="/sessions", tags=["sessions"])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _session_or_404(session_id: int, db: Session):
    s = get_session_by_id(db, session_id)
    if not s:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy session id={session_id}")
    return s


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Bắt đầu phiên khám mới",
)
def start_session(
    body: SessionCreate,
    db:   Session = Depends(get_db),
):
    # Kiểm tra bệnh nhân tồn tại
    patient = get_patient(db, body.patient_id)
    if not patient:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy bệnh nhân id={body.patient_id}",
        )

    session = create_session(db, body)
    return SessionResponse(
        id=session.id, patient_id=session.patient_id,
        status=session.status, notes=session.notes,
        started_at=session.started_at, ended_at=session.ended_at,
        duration_sec=session.duration_sec,
    )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Chi tiết phiên khám",
    responses={404: {"model": ErrorResponse}},
)
def get_session(session_id: int, db: Session = Depends(get_db)):
    s = _session_or_404(session_id, db)
    return SessionResponse(
        id=s.id, patient_id=s.patient_id,
        status=s.status, notes=s.notes,
        started_at=s.started_at, ended_at=s.ended_at,
        duration_sec=s.duration_sec,
    )


@router.patch(
    "/{session_id}/end",
    response_model=SessionDiagnosis,
    summary="Kết thúc phiên — tổng hợp chẩn đoán",
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
def end_session_and_diagnose(
    session_id: int,
    body:       SessionEnd    = SessionEnd(),
    db:         Session       = Depends(get_db),
):
    """
    Kết thúc phiên khám và trả về chẩn đoán tổng hợp từ tất cả window.
    Lưu SessionDiagnosis vào bảng reports.
    """
    s = _session_or_404(session_id, db)
    if s.status != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Session id={session_id} đã ở trạng thái '{s.status}'",
        )

    # Lấy tất cả predictions đã lưu
    pred_rows = get_predictions_by_session(db, session_id)
    if not pred_rows:
        raise HTTPException(
            status_code=400,
            detail="Không có dữ liệu predict — hãy đo ít nhất 1 window trước khi kết thúc",
        )

    # Chuyển ORM rows → Pydantic WindowPrediction
    predictions = [
        WindowPrediction(
            window_id     = r.window_id,
            label         = r.label,
            label_index   = r.label_index,
            confidence    = r.confidence,
            is_uncertain  = r.is_uncertain,
            probabilities = {
                "Normal":    r.prob_normal,
                "Parkinson": r.prob_parkinson,
                "Abnormal":  r.prob_abnormal,
            },
        )
        for r in pred_rows
    ]

    # Tổng hợp chẩn đoán
    diagnosis = ml_service.finalize_session(session_id, predictions)

    # Đóng session + lưu report
    end_session(db, session_id, notes=body.notes)
    save_report(db, session_id, diagnosis, doctor_notes=body.notes)

    return diagnosis


@router.post(
    "/{session_id}/data",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Ghi 1 window sensor vào InfluxDB",
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
def ingest_sensor_window(
    session_id: int,
    window:     SensorWindow,
    db:         Session          = Depends(get_db),
    influx:     InfluxDBService  = Depends(get_influx),
):
    """
    Frontend gọi endpoint này sau mỗi window (mỗi 2 giây).
    Data ghi thẳng vào InfluxDB — không qua PostgreSQL.
    """
    s = _session_or_404(session_id, db)
    if s.status != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Session id={session_id} đã kết thúc, không nhận data mới",
        )

    influx.write_sensor_window(
        session_id = session_id,
        window_id  = window.window_id,
        timestamp  = window.timestamp,
        s1_values  = window.s1,
        s2_values  = window.s2,
        s3_values  = window.s3,
        s4_values  = window.s4,
    )
    # 204 No Content — không trả body


@router.get(
    "/{session_id}/predictions",
    response_model=SessionPredictionHistory,
    summary="Lịch sử predict của phiên",
    responses={404: {"model": ErrorResponse}},
)
def get_session_predictions(session_id: int, db: Session = Depends(get_db)):
    _session_or_404(session_id, db)
    rows = get_predictions_by_session(db, session_id)

    predictions = [
        WindowPrediction(
            window_id     = r.window_id,
            label         = r.label,
            label_index   = r.label_index,
            confidence    = r.confidence,
            is_uncertain  = r.is_uncertain,
            probabilities = {
                "Normal":    r.prob_normal,
                "Parkinson": r.prob_parkinson,
                "Abnormal":  r.prob_abnormal,
            },
        )
        for r in rows
    ]

    return SessionPredictionHistory(
        session_id  = session_id,
        total       = len(predictions),
        predictions = predictions,
        diagnosis   = None,   # None nếu session chưa kết thúc
    )


@router.get(
    "/{session_id}/diagnosis",
    response_model=SessionDiagnosis,
    summary="Chẩn đoán tổng hợp của phiên đã kết thúc",
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
def get_session_diagnosis(session_id: int, db: Session = Depends(get_db)):
    s = _session_or_404(session_id, db)
    if not s.report:
        raise HTTPException(
            status_code=400,
            detail="Session chưa có báo cáo — gọi PATCH /sessions/{id}/end trước",
        )
    r = s.report
    return SessionDiagnosis(
        diagnosis         = r.diagnosis,
        label_index       = next(
            (i for i, n in enumerate(["Normal", "Parkinson", "Abnormal"])
             if n == r.diagnosis), -1
        ),
        confidence_mean   = r.confidence_mean,
        vote_distribution = {},   # không lưu vote distribution trong DB, dùng predictions để tính lại
        total_windows     = r.total_windows,
        confident_windows = round(r.total_windows * r.confident_ratio),
        confident_ratio   = r.confident_ratio,
        sufficient_data   = True,
    )