# 🦶 Smart Insole — Hệ thống phát hiện Parkinson qua lót giày thông minh

Hệ thống thu thập dữ liệu áp lực bàn chân (4× FSR402) qua BLE từ ESP32C3, phân tích real-time bằng mô hình LSTM, và hiển thị kết quả chẩn đoán trên giao diện web tại phòng khám.

---

## Kiến trúc hệ thống

```
[ESP32C3 + 4× FSR] ──BLE──► [Trình duyệt bác sĩ]
                                      │
                              Web Bluetooth API
                                      │
                              [React Frontend]
                                      │
                              WebSocket / REST
                                      │
                           ┌─── [FastAPI Backend] ───┐
                           │                         │
                      [PostgreSQL]             [InfluxDB]
                    patients, sessions        sensor time-series
                           │
                      [LSTM Model]
                    inference 50Hz
```

| Thành phần | Công nghệ |
|---|---|
| Firmware | Arduino (Seeed XIAO ESP32C3), NimBLE, ADS1115 |
| Backend | FastAPI, SQLAlchemy, InfluxDB client, TensorFlow |
| Frontend | React 18, Vite, Chart.js, Web Bluetooth API |
| ML | TensorFlow/Keras LSTM, scikit-learn, scipy |
| Infra | Docker Compose, Nginx, PostgreSQL 16, InfluxDB 2.7 |

---

## Yêu cầu

- Docker & Docker Compose
- Python 3.10 hoặc 3.11 (nếu chạy ML local)
- Node.js 18+ (nếu dev frontend local)
- Arduino IDE 2.x + board Seeed XIAO ESP32C3 (nếu flash firmware)
- Chrome hoặc Edge (Web Bluetooth API)

---

## Cài đặt & Chạy

### 1. Clone repo và tạo file `.env`

```bash
git clone <repo-url>
cd Smart\ Insole

cp .env.example .env
# Mở .env và chỉnh các giá trị nếu cần (token, password, model path...)
```

### 2. Train model ML (lần đầu)

> Bỏ qua bước này nếu đã có file `.h5` trong `ml/models/`.

```bash
# Tạo venv và cài dependencies
cd ml
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

# Chạy theo thứ tự:
python src/preprocess.py         # tạo X_train.npy, y_train.npy, ...
python src/train.py              # tạo ml/models/lstm_*.h5
python src/evaluate.py           # xem confusion matrix, ROC curves
```

Sau khi train xong, cập nhật `MODEL_PATH` trong `.env` trỏ vào file `.h5` vừa tạo.

### 3. Build frontend

```bash
cd frontend
npm install
npm run build                    # tạo frontend/dist/
```

### 4. Khởi động toàn bộ hệ thống

```bash
# Từ thư mục gốc Smart Insole/
docker-compose up --build
```

| Service | URL |
|---|---|
| Giao diện bác sĩ | http://localhost |
| FastAPI Swagger | http://localhost:8000/docs |
| InfluxDB UI | http://localhost:8086 |
| Health check | http://localhost:8000/health |

Dừng và xóa data:
```bash
docker-compose down -v
```

---

## Cấu trúc thư mục

```
Smart Insole/
├── .env.example              # Mẫu biến môi trường — copy thành .env
├── .gitignore
├── docker-compose.yml
├── nginx/
│   └── nginx.conf
│
├── firmware/
│   └── smart_insole/
│       └── smart_insole.ino  # Flash lên Seeed XIAO ESP32C3
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py           # FastAPI entry point
│       ├── config.py         # Đọc biến từ .env
│       ├── models/schemas.py # Pydantic schemas
│       ├── routers/          # patients, sessions, predictions, websocket
│       └── services/
│           ├── db_service.py # PostgreSQL + InfluxDB
│           └── ml_service.py # Load model, chạy inference
│
├── frontend/
│   ├── vite.config.js
│   ├── src/
│   │   ├── App.jsx
│   │   ├── pages/            # Dashboard, Examination, Report
│   │   ├── components/       # BLEConnect, SensorChart, ResultDisplay, ...
│   │   ├── hooks/useBLE.js   # Web Bluetooth logic
│   │   └── services/         # apiService.js, bleServices.js
│   └── dist/                 # Build output (gitignored)
│
└── ml/
    ├── requirements.txt
    ├── data/
    │   ├── raw/              # Dataset gốc GaitPDB/GaitNDD (gitignored)
    │   └── processed/        # .npy đã preprocess (gitignored)
    ├── models/               # .h5, .pkl, báo cáo train (gitignored)
    ├── notebooks/            # 01_explore, 02_preprocess, 03_train
    └── src/
        ├── preprocess.py
        ├── model.py
        ├── train.py
        ├── evaluate.py
        └── inference.py
```

---

## Workflow phòng khám

1. Bác sĩ mở trình duyệt → http://localhost
2. **Tạo bệnh nhân** mới hoặc tìm bệnh nhân cũ
3. **Bắt đầu phiên khám** → hệ thống tạo session ID
4. **Kết nối BLE** → chọn thiết bị `SmartInsole` qua Web Bluetooth
5. Bệnh nhân đi bộ → dữ liệu FSR stream 50Hz qua WebSocket
6. Kết quả chẩn đoán từng window hiển thị real-time
7. **Kết thúc phiên** → xem báo cáo tổng hợp

---

## Hardware — Sơ đồ chân ESP32C3

| Cảm biến | Vị trí | Chân | ADC |
|---|---|---|---|
| S1 | Ụ ngón cái | GPIO4 (D2) | ESP32 12-bit |
| S2 | Ngón cái | ADS1115 ch0 | 16-bit |
| S3 | Ụ ngón út | GPIO2 (D0) | ESP32 12-bit |
| S4 | Gót chân | GPIO3 (D1) | ESP32 12-bit |

ADS1115: SDA→D4 (GPIO6), SCL→D5 (GPIO7), VDD→3.3V, ADDR→GND (0x48)

### Thư viện Arduino cần cài
- **NimBLE-Arduino** (by h2zero)
- **Adafruit ADS1X15** (by Adafruit)

---

## API nhanh

```
GET  /health                          # Trạng thái hệ thống
POST /api/v1/patients                 # Tạo bệnh nhân
POST /api/v1/sessions                 # Bắt đầu phiên khám
WS   /api/v1/ws/session/{id}         # Stream sensor data
PATCH /api/v1/sessions/{id}/end      # Kết thúc phiên
GET  /api/v1/sessions/{id}/diagnosis # Xem báo cáo
```

Xem đầy đủ tại: http://localhost:8000/docs

---

## Biến môi trường quan trọng

| Biến | Mô tả |
|---|---|
| `POSTGRES_URL` | Connection string PostgreSQL |
| `INFLUXDB_URL` | URL InfluxDB service |
| `INFLUXDB_TOKEN` | API token InfluxDB |
| `MODEL_PATH` | Đường dẫn file `.h5` đã train |
| `CONFIDENCE_THRESHOLD` | Ngưỡng confidence tối thiểu (mặc định: 0.70) |
| `WS_BUFFER_SIZE` | Số sample mỗi window (mặc định: 100 = 2s ở 50Hz) |

Xem toàn bộ trong `.env.example`.

---

## Dev local (không Docker)

```bash
# Backend
cd Smart\ Insole/
python -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (terminal khác)
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

Lưu ý: cần PostgreSQL và InfluxDB đang chạy (có thể dùng Docker chỉ cho 2 service đó):
```bash
docker-compose up postgres influxdb
```