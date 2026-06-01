"""
config.py — Cấu hình tập trung
================================
Đọc toàn bộ biến môi trường từ .env qua pydantic-settings.
Tất cả file khác import settings từ đây — không hardcode ở bất kỳ đâu.

Dùng:
    from app.config import settings

    db_url = settings.POSTGRES_URL
    token  = settings.INFLUXDB_TOKEN
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Mỗi field tương ứng 1 biến trong .env (hoặc biến môi trường hệ thống).
    Pydantic-settings tự đọc, parse, và validate — raise lỗi rõ ràng
    nếu biến bắt buộc bị thiếu.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,       # POSTGRES_URL khác postgres_url
        extra="ignore",            # bỏ qua biến .env không khai báo ở đây
    )

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------
    POSTGRES_URL: str = Field(
        default="postgresql://smartinsole:password@localhost:5432/smartinsole",
        description="Connection string SQLAlchemy cho PostgreSQL",
        examples=["postgresql://user:pass@postgres:5432/smartinsole"],
    )

    # ------------------------------------------------------------------
    # InfluxDB
    # ------------------------------------------------------------------
    INFLUXDB_URL: str = Field(
        default="http://localhost:8086",
        description="URL của InfluxDB instance",
    )
    INFLUXDB_TOKEN: str = Field(
        default="dev-token-change-in-production",
        description="API token của InfluxDB (tạo trong InfluxDB UI)",
    )
    INFLUXDB_ORG: str = Field(
        default="smartinsole",
        description="Tên organization trong InfluxDB",
    )
    INFLUXDB_BUCKET: str = Field(
        default="sensor_data",
        description="Bucket lưu time-series sensor readings",
    )

    # ------------------------------------------------------------------
    # ML Model
    # ------------------------------------------------------------------
    MODEL_PATH: Path = Field(
        default=Path("ml/models/lstm_v1.h5"),
        description="Đường dẫn tới file model .h5 đã train",
    )
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Ngưỡng confidence tối thiểu để accept prediction",
    )
    MIN_CONFIDENT_RATIO: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Tỉ lệ window confident tối thiểu để xuất chẩn đoán",
    )

    # ------------------------------------------------------------------
    # FastAPI / CORS
    # ------------------------------------------------------------------
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Danh sách origin được phép gọi API (Web Bluetooth chạy trên Chrome)",
    )
    API_PREFIX: str     = Field(default="/api/v1")
    DEBUG:      bool    = Field(default=False)
    APP_TITLE:  str     = Field(default="Smart Insole API")
    APP_VERSION: str    = Field(default="1.0.0")

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------
    WS_BUFFER_SIZE: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Số sample tích lũy trước khi tạo window và chạy predict (= WINDOW_SIZE)",
    )
    WS_MAX_CONNECTIONS: int = Field(
        default=10,
        description="Số WebSocket connections đồng thời tối đa",
    )

    # ------------------------------------------------------------------
    # Sensor hardware
    # ------------------------------------------------------------------
    ADC_ADS1115_MAX: int = Field(
        default=32767,
        description="Giá trị ADC tối đa của ADS1115 (16-bit)",
    )
    ADC_ESP32_MAX: int = Field(
        default=4095,
        description="Giá trị ADC tối đa của ESP32 nội (12-bit)",
    )
    SENSOR_SAMPLE_RATE_HZ: int = Field(
        default=50,
        description="Tần số lấy mẫu từ firmware (Hz)",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("MODEL_PATH")
    @classmethod
    def model_path_must_exist_or_warn(cls, v: Path) -> Path:
        """
        Không raise lỗi nếu file chưa có (chưa train xong).
        Chỉ cảnh báo — để backend vẫn start được khi chưa có model.
        """
        if not v.exists():
            import warnings
            warnings.warn(
                f"MODEL_PATH '{v}' không tồn tại. "
                "InferenceEngine sẽ lỗi khi load — chạy train.py trước.",
                stacklevel=2,
            )
        return v

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v):
        """
        Cho phép khai báo CORS_ORIGINS trong .env theo 2 cách:
            CORS_ORIGINS=http://localhost:3000,http://192.168.1.10:3000
            CORS_ORIGINS=["http://localhost:3000","http://192.168.1.10:3000"]
        """
        if isinstance(v, str):
            # Thử parse JSON array trước
            stripped = v.strip()
            if stripped.startswith("["):
                import json
                return json.loads(stripped)
            # Fallback: comma-separated
            return [x.strip() for x in stripped.split(",") if x.strip()]
        return v

    # ------------------------------------------------------------------
    # Computed helpers (không đọc từ .env)
    # ------------------------------------------------------------------

    @property
    def influxdb_config(self) -> dict:
        """Dict truyền thẳng vào InfluxDB client constructor."""
        return {
            "url":   self.INFLUXDB_URL,
            "token": self.INFLUXDB_TOKEN,
            "org":   self.INFLUXDB_ORG,
        }

    @property
    def is_production(self) -> bool:
        return not self.DEBUG

    @property
    def window_duration_sec(self) -> float:
        """Thời gian thực của 1 window (giây)."""
        return self.WS_BUFFER_SIZE / self.SENSOR_SAMPLE_RATE_HZ


# ---------------------------------------------------------------------------
# Singleton — dùng lru_cache để chỉ đọc .env 1 lần
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Trả về Settings singleton.
    FastAPI dependency injection dùng hàm này:

        from app.config import get_settings
        from fastapi import Depends

        @router.get("/info")
        def info(settings: Settings = Depends(get_settings)):
            return {"version": settings.APP_VERSION}

    Hoặc dùng trực tiếp ở module level:
        from app.config import settings
    """
    return Settings()


# Module-level singleton — tiện dùng khi không cần DI
settings: Settings = get_settings()


# ---------------------------------------------------------------------------
# Kiểm tra nhanh
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Kiểm tra config.py ===\n")

    s = Settings()

    print(f"POSTGRES_URL       : {s.POSTGRES_URL}")
    print(f"INFLUXDB_URL       : {s.INFLUXDB_URL}")
    print(f"INFLUXDB_BUCKET    : {s.INFLUXDB_BUCKET}")
    print(f"MODEL_PATH         : {s.MODEL_PATH}")
    print(f"CONFIDENCE_THRESH  : {s.CONFIDENCE_THRESHOLD}")
    print(f"CORS_ORIGINS       : {s.CORS_ORIGINS}")
    print(f"DEBUG              : {s.DEBUG}")
    print(f"window_duration    : {s.window_duration_sec:.1f}s")
    print(f"is_production      : {s.is_production}")
    print(f"influxdb_config    : {s.influxdb_config}")

    # Kiểm tra parse CORS từ string
    import os
    os.environ["CORS_ORIGINS"] = "http://localhost:3000,http://192.168.1.5:3000"
    get_settings.cache_clear()
    s2 = get_settings()
    assert len(s2.CORS_ORIGINS) == 2
    assert "http://localhost:3000" in s2.CORS_ORIGINS
    print("\nCORS comma-string parse: OK —", s2.CORS_ORIGINS)

    # Kiểm tra singleton
    s3 = get_settings()
    assert s2 is s3, "get_settings() phải trả về cùng 1 object"
    print("Singleton (lru_cache): OK")

    print("\n=== Tất cả test PASSED ===")