# ============================================================
# Smart Insole — Backend Dockerfile
# ============================================================
# Build context: thư mục gốc Smart Insole/ (docker-compose.yml)
# Vì thế các COPY path đều bắt đầu từ Smart Insole/:
#   COPY backend/... → đúng
#   COPY ml/...      → đúng (nếu cần)
#
# Cấu trúc bên trong container:
#   /app/
#   ├── backend/          ← code FastAPI (COPY + volume mount dev)
#   │   └── app/
#   │       ├── main.py
#   │       ├── config.py
#   │       ├── models/
#   │       ├── routers/
#   │       └── services/
#   │           └── ml_service.py
#   └── ml/               ← model + src (volume mount read-only)
#       ├── models/
#       │   └── lstm_v1.h5
#       └── src/
#           ├── inference.py
#           ├── preprocess.py
#           └── model.py
#
# WORKDIR = /app (không phải /app/backend)
# Lý do: MODEL_PATH = "ml/models/lstm_v1.h5" trong .env là relative từ /app
#         → /app/ml/models/lstm_v1.h5 ✅
#
# PYTHONPATH = /app/backend
# Lý do: cho phép "from app.config import settings" resolve đúng
#         → /app/backend/app/config.py ✅
# ============================================================

FROM python:3.11-slim

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
# gcc         : biên dịch các extension C (một số numpy/scipy ops)
# libpq-dev   : header PostgreSQL cho psycopg2
# curl        : dùng trong Docker HEALTHCHECK
# ca-certificates: HTTPS calls (InfluxDB, download nếu cần)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Working directory
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# Cài Python dependencies trước khi copy code
# → Docker cache layer — chỉ rebuild khi requirements.txt thay đổi
# ------------------------------------------------------------
COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/backend/requirements.txt

# ------------------------------------------------------------
# Copy backend code vào image
# (bị override bởi volume mount khi dev, dùng thật khi production)
# ------------------------------------------------------------
COPY backend/ /app/backend/

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

# PYTHONPATH: "from app.config import" → /app/backend/app/config.py
ENV PYTHONPATH=/app/backend

# Không tạo file .pyc (giảm size image, tránh stale cache)
ENV PYTHONDONTWRITEBYTECODE=1

# Log ra stdout ngay lập tức, không buffer (quan trọng cho Docker logs)
ENV PYTHONUNBUFFERED=1

# TensorFlow: tắt GPU warning nếu không có GPU (server phòng khám thường không có)
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""

# ------------------------------------------------------------
# Port
# ------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------
# Docker dùng để tự restart container nếu /health trả lỗi
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ------------------------------------------------------------
# Startup command
# ------------------------------------------------------------
# --host 0.0.0.0 : nhận kết nối từ ngoài container
# --port 8000    : khớp với EXPOSE và docker-compose ports
# --workers 1    : 1 worker vì TensorFlow model không share được giữa processes
#                  (InferenceEngine singleton chỉ an toàn với 1 process)
# Khi dev (docker-compose mount ./backend:/app/backend):
#   thêm --reload vào CMD hoặc dùng override trong docker-compose
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]

# ============================================================
# Ghi chú: vì sao --workers 1
# ============================================================
# uvicorn --workers N sẽ fork N process.
# InferenceEngine dùng singleton + threading.Lock → thread-safe.
# Nhưng KHÔNG process-safe: mỗi process load TF model riêng → tốn RAM gấp N lần.
# Với PC phòng khám (8-16GB RAM, 1 bác sĩ dùng cùng lúc) → 1 worker là đủ.
# Nếu cần scale: dùng gunicorn + uvicorn worker class thay vì --workers.
# ============================================================