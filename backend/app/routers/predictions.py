"""
predictions.py — Router ML Inference
======================================
POST /api/v1/predict          ← nhận 1 window → predict ngay → lưu DB
GET  /api/v1/model/info       ← thông tin model đang chạy
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    ErrorResponse,
)
from app.services.db_service import (
    get_db,
    get_session_by_id,
    save_prediction,
)
from app.services.ml_service import ml_service

router = APIRouter(tags=["predictions"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict 1 window sensor",
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse, "description": "Model chưa sẵn sàng"},
    },
)
def predict_window(
    body: PredictionRequest,
    db:   Session = Depends(get_db),
):
    """
    Nhận 1 SensorWindow (100 timestep × 4 sensor) từ frontend,
    chạy LSTM inference, lưu kết quả vào PostgreSQL.

    Frontend gọi endpoint này sau mỗi 2 giây (mỗi window).
    Kết quả hiển thị real-time trên SensorChart.
    """
    # Kiểm tra model sẵn sàng
    if not ml_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="ML model chưa sẵn sàng. Kiểm tra logs để biết thêm.",
        )

    # Kiểm tra session tồn tại và đang active
    session = get_session_by_id(db, body.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy session id={body.session_id}",
        )
    if session.status != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Session id={body.session_id} đã kết thúc",
        )

    # Kiểm tra window_id khớp session_id
    if body.window.session_id != body.session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id trong body và window không khớp",
        )

    # Predict
    prediction, latency_ms = ml_service.predict_window(body.window)

    # Lưu vào PostgreSQL
    saved = False
    try:
        save_prediction(db, session_id=body.session_id, pred=prediction)
        saved = True
    except Exception as e:
        # Không crash nếu DB write lỗi — prediction vẫn trả về cho frontend
        import logging
        logging.getLogger(__name__).error("Lỗi lưu prediction vào DB: %s", e)

    return PredictionResponse(
        session_id        = body.session_id,
        window_prediction = prediction,
        saved_to_db       = saved,
    )


@router.get(
    "/model/info",
    response_model=ModelInfo,
    summary="Thông tin model đang chạy",
)
def get_model_info():
    """
    Trả về metadata của LSTM model: accuracy, AUC, số parameters, threshold.
    Dùng để hiển thị trong dashboard của bác sĩ.
    """
    info = ml_service.model_info
    return ModelInfo(
        model_path           = info.get("model_path", ""),
        confidence_threshold = info.get("confidence_threshold", 0.70),
        classes              = info.get("classes", []),
        input_shape          = info.get("input_shape", []),
        n_parameters         = info.get("n_parameters", 0),
        eval_accuracy        = info.get("eval_accuracy"),
        eval_roc_auc         = info.get("eval_roc_auc"),
        eval_macro_f1        = info.get("eval_macro_f1"),
    )