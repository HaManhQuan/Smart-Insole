"""
patients.py — Router quản lý bệnh nhân
========================================
GET  /api/v1/patients            ← danh sách (phân trang + tìm kiếm)
POST /api/v1/patients            ← tạo mới
GET  /api/v1/patients/{id}       ← chi tiết 1 bệnh nhân
GET  /api/v1/patients/{id}/sessions  ← lịch sử phiên khám
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.models.schemas import (
    PatientCreate,
    PatientListResponse,
    PatientResponse,
    SessionResponse,
    ErrorResponse,
)
from app.services.db_service import (
    get_db,
    create_patient,
    get_patient,
    list_patients,
    list_sessions_by_patient,
)

router = APIRouter(prefix="/patients", tags=["patients"])


@router.post(
    "",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Tạo bệnh nhân mới",
)
def create_new_patient(
    body: PatientCreate,
    db: Session = Depends(get_db),
):
    patient = create_patient(db, body)
    return patient


@router.get(
    "",
    response_model=PatientListResponse,
    summary="Danh sách bệnh nhân",
)
def get_patients(
    page:     int           = Query(default=1,  ge=1),
    per_page: int           = Query(default=20, ge=1, le=100),
    search:   str | None    = Query(default=None, description="Tìm theo tên"),
    db:       Session       = Depends(get_db),
):
    items, total = list_patients(db, page=page, per_page=per_page, search=search)
    return PatientListResponse(total=total, page=page, per_page=per_page, items=items)


@router.get(
    "/{patient_id}",
    response_model=PatientResponse,
    summary="Chi tiết bệnh nhân",
    responses={404: {"model": ErrorResponse}},
)
def get_patient_detail(
    patient_id: int,
    db: Session = Depends(get_db),
):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy bệnh nhân id={patient_id}")
    return patient


@router.get(
    "/{patient_id}/sessions",
    response_model=list[SessionResponse],
    summary="Lịch sử phiên khám của bệnh nhân",
    responses={404: {"model": ErrorResponse}},
)
def get_patient_sessions(
    patient_id: int,
    limit: int  = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy bệnh nhân id={patient_id}")

    sessions = list_sessions_by_patient(db, patient_id=patient_id, limit=limit)

    # Thêm duration_sec (computed property không có trong ORM column)
    results = []
    for s in sessions:
        d = {
            "id":           s.id,
            "patient_id":   s.patient_id,
            "status":       s.status,
            "notes":        s.notes,
            "started_at":   s.started_at,
            "ended_at":     s.ended_at,
            "duration_sec": s.duration_sec,
        }
        results.append(SessionResponse(**d))
    return results