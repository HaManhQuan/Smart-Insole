"""
main.py — FastAPI Entry Point
===============================
Chạy local:
    uvicorn app.main:app --reload --port 8000

Chạy qua Docker:
    docker-compose up

Swagger UI: http://localhost:8000/docs
ReDoc     : http://localhost:8000/redoc
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schemas import ErrorResponse
from app.services.db_service import (
    create_tables,
    init_influx,
    close_influx,
    check_postgres_health,
    check_influx_health,
    get_db,
)
from app.routers import patients, sessions, predictions, websocket

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Thay thế @app.on_event("startup/shutdown") — cách mới của FastAPI.
    Code trước yield = startup, code sau yield = shutdown.
    """
    log.info("=== Smart Insole API starting ===")

    # 1. Tạo bảng PostgreSQL nếu chưa có
    try:
        create_tables()
        log.info("PostgreSQL: OK")
    except Exception as e:
        log.error("PostgreSQL init failed: %s", e)
        raise

    # 2. Kết nối InfluxDB
    try:
        init_influx()
        log.info("InfluxDB: OK")
    except Exception as e:
        log.warning("InfluxDB init failed (non-fatal): %s", e)
        # Không raise — app vẫn chạy được nếu InfluxDB chưa sẵn sàng

    # 3. Warm-up ML model (lazy load — lỗi ở đây chỉ warning, không crash)
    try:
        from app.services.ml_service import ml_service
        if ml_service.is_ready():
            log.info("ML model: OK — %s", settings.MODEL_PATH)
        else:
            log.warning("ML model: chưa sẵn sàng — chạy train.py để tạo model")
    except Exception as e:
        log.warning("ML model load failed (non-fatal): %s", e)

    log.info("=== API ready — http://localhost:8000/docs ===")

    yield   # ← app đang chạy

    # Shutdown
    log.info("=== Smart Insole API shutting down ===")
    close_influx()
    log.info("InfluxDB connection closed")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title        = settings.APP_TITLE,
    version      = settings.APP_VERSION,
    description  = """
## Smart Insole — Parkinson Detection API

Backend cho hệ thống phát hiện Parkinson qua lót giày thông minh.

### Workflow tại phòng khám
1. **Tạo bệnh nhân** → `POST /api/v1/patients`
2. **Bắt đầu phiên khám** → `POST /api/v1/sessions`
3. **Kết nối BLE** → WebSocket `/api/v1/ws/session/{id}`
4. **Stream sensor data** → 50Hz qua WebSocket
5. **Nhận kết quả real-time** → `window_result` messages
6. **Kết thúc phiên** → `PATCH /api/v1/sessions/{id}/end`
7. **Xem báo cáo** → `GET /api/v1/sessions/{id}/diagnosis`
""",
    docs_url     = "/docs",
    redoc_url    = "/redoc",
    lifespan     = lifespan,
)


# ---------------------------------------------------------------------------
# CORS — cho phép Web Bluetooth từ Chrome/Edge
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.CORS_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled exception: %s %s → %s", request.method, request.url, exc)
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content     = ErrorResponse(
            error  = "Internal server error",
            detail = str(exc) if settings.DEBUG else None,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(patients.router,   prefix=settings.API_PREFIX)
app.include_router(sessions.router,   prefix=settings.API_PREFIX)
app.include_router(predictions.router, prefix=settings.API_PREFIX)
app.include_router(websocket.router,  prefix=settings.API_PREFIX)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["system"],
    summary="Kiểm tra trạng thái hệ thống",
)
def health_check():
    """
    Trả về trạng thái của tất cả services.
    Dùng cho Docker HEALTHCHECK và monitoring.
    """
    from app.services.ml_service import ml_service

    db_gen = get_db()
    db = next(db_gen)
    try:
        pg_status     = check_postgres_health(db)
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass

    influx_status = check_influx_health()
    ml_ready      = ml_service.is_ready()

    overall = (
        pg_status["status"] == "ok"
        and influx_status["status"] == "ok"
        and ml_ready
    )

    return {
        "status":     "ok" if overall else "degraded",
        "version":    settings.APP_VERSION,
        "services": {
            "postgres": pg_status,
            "influxdb": influx_status,
            "ml_model": {"status": "ok" if ml_ready else "unavailable"},
        },
    }


@app.get("/", tags=["system"], include_in_schema=False)
def root():
    return {
        "name":    settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "docs":    "/docs",
    }