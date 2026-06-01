"""
db_service.py — Database Layer
================================
Quản lý 2 database:
  - PostgreSQL (SQLAlchemy)  : patients, sessions, predictions, reports
  - InfluxDB                 : sensor time-series readings (50Hz)

Import trong routers:
    from app.services.db_service import (
        get_db, create_patient, get_session_by_id, write_sensor_batch, ...
    )
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Optional

# SQLAlchemy
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, create_engine, text,
)
from sqlalchemy.orm import (
    DeclarativeBase, Session, relationship, sessionmaker,
)

# InfluxDB
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from app.config import settings
from app.models.schemas import (
    PatientCreate, SessionCreate,
    SessionDiagnosis, WindowPrediction,
)

log = logging.getLogger(__name__)


# ===========================================================================
# PostgreSQL — SQLAlchemy ORM
# ===========================================================================

# ---------------------------------------------------------------------------
# Engine & Session factory
# ---------------------------------------------------------------------------

engine = create_engine(
    settings.POSTGRES_URL,
    pool_size=5,            # connection pool — đủ cho phòng khám 1–2 bác sĩ
    max_overflow=10,
    pool_pre_ping=True,     # kiểm tra connection còn sống trước khi dùng
    echo=settings.DEBUG,    # log SQL khi DEBUG=True
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# ORM Models (ánh xạ tới bảng PostgreSQL)
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class PatientORM(Base):
    __tablename__ = "patients"

    id         = Column(Integer, primary_key=True, index=True)
    full_name  = Column(String(120), nullable=False)
    birth_year = Column(Integer, nullable=False)
    gender     = Column(String(10), nullable=False)
    phone      = Column(String(20), nullable=True)
    notes      = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    sessions = relationship("SessionORM", back_populates="patient", lazy="dynamic")


class SessionORM(Base):
    __tablename__ = "sessions"

    id         = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    status     = Column(String(20), nullable=False, default="active")
    notes      = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    ended_at   = Column(DateTime(timezone=True), nullable=True)

    patient     = relationship("PatientORM", back_populates="sessions")
    predictions = relationship("PredictionORM", back_populates="session", lazy="dynamic")
    report      = relationship("ReportORM", back_populates="session", uselist=False)

    @property
    def duration_sec(self) -> Optional[float]:
        if self.ended_at and self.started_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None


class PredictionORM(Base):
    __tablename__ = "predictions"

    id            = Column(Integer, primary_key=True, index=True)
    session_id    = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)
    window_id     = Column(Integer, nullable=False)
    label         = Column(String(20), nullable=False)
    label_index   = Column(Integer, nullable=False)
    confidence    = Column(Float, nullable=False)
    is_uncertain  = Column(Boolean, nullable=False, default=False)
    prob_normal   = Column(Float, nullable=False)
    prob_parkinson = Column(Float, nullable=False)
    prob_abnormal = Column(Float, nullable=False)
    created_at    = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("SessionORM", back_populates="predictions")


class ReportORM(Base):
    __tablename__ = "reports"

    id               = Column(Integer, primary_key=True, index=True)
    session_id       = Column(Integer, ForeignKey("sessions.id"), nullable=False, unique=True)
    diagnosis        = Column(String(20), nullable=False)
    confidence_mean  = Column(Float, nullable=False)
    confident_ratio  = Column(Float, nullable=False)
    total_windows    = Column(Integer, nullable=False)
    doctor_notes     = Column(Text, nullable=True)
    pdf_path         = Column(String(300), nullable=True)
    created_at       = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("SessionORM", back_populates="report")


# ---------------------------------------------------------------------------
# Tạo bảng (gọi khi startup)
# ---------------------------------------------------------------------------

def create_tables() -> None:
    """Tạo tất cả bảng nếu chưa tồn tại. Gọi 1 lần trong main.py startup."""
    Base.metadata.create_all(bind=engine)
    log.info("PostgreSQL tables ready")


# ---------------------------------------------------------------------------
# Dependency Injection cho FastAPI
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency — inject DB session vào router:

        @router.get("/patients")
        def list_patients(db: Session = Depends(get_db)):
            return db.query(PatientORM).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_context() -> Generator[Session, None, None]:
    """Context manager cho code không dùng FastAPI DI (ví dụ script thuần Python)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ===========================================================================
# CRUD — Patients
# ===========================================================================

def create_patient(db: Session, data: PatientCreate) -> PatientORM:
    patient = PatientORM(**data.model_dump())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    log.info("Created patient id=%d name=%s", patient.id, patient.full_name)
    return patient


def get_patient(db: Session, patient_id: int) -> Optional[PatientORM]:
    return db.query(PatientORM).filter(PatientORM.id == patient_id).first()


def list_patients(
    db: Session,
    page: int = 1,
    per_page: int = 20,
    search: Optional[str] = None,
) -> tuple[list[PatientORM], int]:
    """Trả về (items, total) cho phân trang."""
    q = db.query(PatientORM)
    if search:
        q = q.filter(PatientORM.full_name.ilike(f"%{search}%"))
    total = q.count()
    items = q.order_by(PatientORM.id.desc()).offset((page - 1) * per_page).limit(per_page).all()
    return items, total


# ===========================================================================
# CRUD — Sessions
# ===========================================================================

def create_session(db: Session, data: SessionCreate) -> SessionORM:
    session = SessionORM(patient_id=data.patient_id, notes=data.notes)
    db.add(session)
    db.commit()
    db.refresh(session)
    log.info("Created session id=%d patient_id=%d", session.id, session.patient_id)
    return session


def get_session_by_id(db: Session, session_id: int) -> Optional[SessionORM]:
    return db.query(SessionORM).filter(SessionORM.id == session_id).first()


def end_session(
    db: Session,
    session_id: int,
    notes: Optional[str] = None,
) -> Optional[SessionORM]:
    session = get_session_by_id(db, session_id)
    if not session:
        return None
    session.status   = "completed"
    session.ended_at = datetime.now(timezone.utc)
    if notes:
        session.notes = notes
    db.commit()
    db.refresh(session)
    log.info("Ended session id=%d duration=%.1fs", session_id, session.duration_sec or 0)
    return session


def list_sessions_by_patient(
    db: Session,
    patient_id: int,
    limit: int = 10,
) -> list[SessionORM]:
    return (
        db.query(SessionORM)
        .filter(SessionORM.patient_id == patient_id)
        .order_by(SessionORM.started_at.desc())
        .limit(limit)
        .all()
    )


# ===========================================================================
# CRUD — Predictions
# ===========================================================================

def save_prediction(
    db: Session,
    session_id: int,
    pred: WindowPrediction,
) -> PredictionORM:
    """Lưu kết quả predict 1 window vào PostgreSQL."""
    row = PredictionORM(
        session_id   = session_id,
        window_id    = pred.window_id,
        label        = pred.label,
        label_index  = pred.label_index,
        confidence   = pred.confidence,
        is_uncertain = pred.is_uncertain,
        prob_normal    = pred.probabilities.get("Normal",    0.0),
        prob_parkinson = pred.probabilities.get("Parkinson", 0.0),
        prob_abnormal  = pred.probabilities.get("Abnormal",  0.0),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_predictions_by_session(
    db: Session,
    session_id: int,
) -> list[PredictionORM]:
    return (
        db.query(PredictionORM)
        .filter(PredictionORM.session_id == session_id)
        .order_by(PredictionORM.window_id)
        .all()
    )


def save_report(
    db: Session,
    session_id: int,
    diagnosis: SessionDiagnosis,
    doctor_notes: Optional[str] = None,
    pdf_path: Optional[str] = None,
) -> ReportORM:
    report = ReportORM(
        session_id      = session_id,
        diagnosis       = diagnosis.diagnosis,
        confidence_mean = diagnosis.confidence_mean,
        confident_ratio = diagnosis.confident_ratio,
        total_windows   = diagnosis.total_windows,
        doctor_notes    = doctor_notes,
        pdf_path        = pdf_path,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    log.info("Saved report id=%d session_id=%d diagnosis=%s",
             report.id, session_id, diagnosis.diagnosis)
    return report


# ===========================================================================
# InfluxDB — Time-series sensor data
# ===========================================================================

class InfluxDBService:
    """
    Wrapper quanh InfluxDB client.
    Dùng SYNCHRONOUS write để đảm bảo data được ghi trước khi response.

    Instantiate 1 lần trong main.py startup:
        influx = InfluxDBService()

    Ghi data:
        influx.write_sensor_window(session_id, window_id, timestamp, s1, s2, s3, s4)

    Query:
        rows = influx.query_session(session_id)
    """

    def __init__(self):
        self._client   = InfluxDBClient(**settings.influxdb_config)
        self._write    = self._client.write_api(write_options=SYNCHRONOUS)
        self._query    = self._client.query_api()
        self._bucket   = settings.INFLUXDB_BUCKET
        self._org      = settings.INFLUXDB_ORG
        log.info("InfluxDB connected: %s / %s", settings.INFLUXDB_URL, self._bucket)

    def write_sensor_window(
        self,
        session_id: int,
        window_id:  int,
        timestamp:  datetime,
        s1_values:  list[float],
        s2_values:  list[float],
        s3_values:  list[float],
        s4_values:  list[float],
        sample_rate_hz: int = 50,
    ) -> None:
        """
        Ghi 1 window (100 samples × 4 sensor) vào InfluxDB.
        Mỗi sample tạo 1 Point với timestamp tính ngược từ cuối window.

        Tag: session_id (để query theo session)
        Fields: s1, s2, s3, s4 (raw ADC values)
        """
        points = []
        interval_us = int(1_000_000 / sample_rate_hz)   # microseconds giữa 2 sample

        ts_us = int(timestamp.timestamp() * 1_000_000)
        offset_start = -len(s1_values) * interval_us    # timestamp đầu tiên = ts - 2s

        for i in range(len(s1_values)):
            sample_ts = ts_us + offset_start + i * interval_us
            p = (
                Point("sensor_reading")
                .tag("session_id", str(session_id))
                .tag("window_id",  str(window_id))
                .field("s1", float(s1_values[i]))
                .field("s2", float(s2_values[i]))
                .field("s3", float(s3_values[i]))
                .field("s4", float(s4_values[i]))
                .time(sample_ts, WritePrecision.MICROSECONDS)
            )
            points.append(p)

        self._write.write(bucket=self._bucket, org=self._org, record=points)
        log.debug("InfluxDB: wrote %d points session=%d window=%d",
                  len(points), session_id, window_id)

    def query_session(
        self,
        session_id: int,
        start: Optional[datetime] = None,
        stop:  Optional[datetime] = None,
    ) -> list[dict]:
        """
        Lấy toàn bộ sensor data của 1 session.
        Dùng cho notebook phân tích hoặc export báo cáo.

        Trả về list[dict] với keys: time, s1, s2, s3, s4.
        """
        start_str = start.isoformat() if start else "-30d"
        stop_str  = stop.isoformat()  if stop  else "now()"

        flux = f"""
from(bucket: "{self._bucket}")
  |> range(start: {start_str}, stop: {stop_str})
  |> filter(fn: (r) => r._measurement == "sensor_reading")
  |> filter(fn: (r) => r.session_id == "{session_id}")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
"""
        tables = self._query.query(flux, org=self._org)
        rows = []
        for table in tables:
            for record in table.records:
                rows.append({
                    "time": record.get_time(),
                    "s1":   record.values.get("s1"),
                    "s2":   record.values.get("s2"),
                    "s3":   record.values.get("s3"),
                    "s4":   record.values.get("s4"),
                })
        return rows

    def delete_session(self, session_id: int) -> None:
        """Xóa toàn bộ data của 1 session (dùng khi test hoặc GDPR request)."""
        delete_api = self._client.delete_api()
        delete_api.delete(
            start="1970-01-01T00:00:00Z",
            stop=datetime.now(timezone.utc).isoformat(),
            predicate=f'session_id="{session_id}"',
            bucket=self._bucket,
            org=self._org,
        )
        log.info("InfluxDB: deleted session_id=%d", session_id)

    def close(self) -> None:
        self._client.close()
        log.info("InfluxDB connection closed")


# ---------------------------------------------------------------------------
# Singleton InfluxDB service — tạo 1 lần trong main.py
# ---------------------------------------------------------------------------

_influx_service: Optional[InfluxDBService] = None


def get_influx() -> InfluxDBService:
    """
    Trả về singleton InfluxDBService.
    Gọi init_influx() trong main.py startup trước khi dùng.
    """
    global _influx_service
    if _influx_service is None:
        raise RuntimeError(
            "InfluxDB chưa được khởi tạo. "
            "Gọi init_influx() trong FastAPI startup event."
        )
    return _influx_service


def init_influx() -> None:
    """Gọi trong main.py @app.on_event('startup')."""
    global _influx_service
    _influx_service = InfluxDBService()


def close_influx() -> None:
    """Gọi trong main.py @app.on_event('shutdown')."""
    global _influx_service
    if _influx_service:
        _influx_service.close()
        _influx_service = None


# ---------------------------------------------------------------------------
# Health check — dùng trong GET /health
# ---------------------------------------------------------------------------

def check_postgres_health(db: Session) -> dict:
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        log.error("PostgreSQL health check failed: %s", e)
        return {"status": "error", "detail": str(e)}


def check_influx_health() -> dict:
    try:
        client = InfluxDBClient(**settings.influxdb_config)
        ready = client.ping()
        client.close()
        return {"status": "ok" if ready else "unreachable"}
    except Exception as e:
        log.error("InfluxDB health check failed: %s", e)
        return {"status": "error", "detail": str(e)}