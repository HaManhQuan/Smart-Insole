"""
preprocess.py — Smart Insole ML Pipeline
=========================================
Xử lý 2 dataset từ PhysioNet:
  - GaitPDB  : 93 PD + 73 healthy  (physionet.org/content/gaitpdb/1.0.0)
  - GaitNDD  : 15 PD + 16 healthy + Huntington + ALS  (physionet.org/content/gaitndd/1.0.0)

Output cuối cùng: numpy arrays shape (N, 100, 4) sẵn sàng đưa vào LSTM.

Label scheme:
  0 = Normal gait
  1 = Parkinson's disease
  2 = Abnormal gait (Huntington, ALS)

Sensor mapping (4 vùng khớp phần cứng FSR402):
  S1 = ụ ngón cái   (1st metatarsal head)
  S2 = ngón cái      (big toe)
  S3 = ụ ngón út    (5th metatarsal head)
  S4 = gót chân      (heel)
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, decimate
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cấu hình — chỉnh đường dẫn nếu cần
# ---------------------------------------------------------------------------

RAW_GAITPDB = Path("ml/data/raw/gaitpdb")   # thư mục chứa các file .txt của GaitPDB
RAW_GAITNDD = Path("ml/data/raw/gaitndd")   # thư mục chứa các file .txt của GaitNDD
PROCESSED_DIR = Path("ml/data/processed")

WINDOW_SIZE = 100       # số timestep mỗi cửa sổ = 2 giây @ 50Hz
STRIDE = 50             # bước trượt = 1 giây (50% overlap)
LOWPASS_CUTOFF = 20.0   # Hz — cắt nhiễu cao tần
TARGET_HZ = 50          # Hz — tần số đầu ra (GaitPDB gốc là 100Hz)
ORIGINAL_HZ = 100       # Hz — GaitPDB sample rate

# Tỉ lệ split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Bước 1: Butterworth low-pass filter
# ---------------------------------------------------------------------------

def butter_lowpass_filter(signal: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Áp Butterworth low-pass filter lên 1 cột tín hiệu.
    cutoff : tần số cắt (Hz)
    fs     : sample rate thực tế của signal (Hz)
    """
    nyq = fs / 2.0
    if cutoff >= nyq:
        log.warning("Cutoff %.1f >= Nyquist %.1f — bỏ qua filter", cutoff, nyq)
        return signal
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    # filtfilt: zero-phase (không lệch pha)
    return filtfilt(b, a, signal, axis=0)


def apply_filter_to_array(data: np.ndarray, cutoff: float, fs: float) -> np.ndarray:
    """Áp filter lên từng cột của array shape (T, C)."""
    filtered = np.empty_like(data)
    for c in range(data.shape[1]):
        filtered[:, c] = butter_lowpass_filter(data[:, c], cutoff, fs)
    return filtered


# ---------------------------------------------------------------------------
# Bước 2: Aggregate 16 sensor GaitPDB → 4 sensor của bạn
# ---------------------------------------------------------------------------

def aggregate_gaitpdb_columns(df: pd.DataFrame) -> np.ndarray:
    """
    GaitPDB file format (19 cột):
      col 0  : time (seconds)
      col 1–8: Left foot sensors L1–L8 (Newton)
      col 9–16: Right foot sensors R1–R8 (Newton)
      col 17 : Total left
      col 18 : Total right

    Sensor layout GaitPDB (mỗi chân, từ gót đến ngón):
      idx 0,1 → heel (gót)
      idx 2,3 → metatarsal 1–2 (ụ ngón cái phía trong)
      idx 4   → metatarsal 3   (giữa)
      idx 5   → metatarsal 5   (ụ ngón út phía ngoài)
      idx 6,7 → toe (ngón chân)

    Mapping sang 4 sensor của bạn (dùng left foot):
      S1 (ụ ngón cái)  ← mean(L3, L4)   col index 3,4
      S2 (ngón cái)    ← mean(L7, L8)   col index 7,8
      S3 (ụ ngón út)   ← L6             col index 6
      S4 (gót)         ← mean(L1, L2)   col index 1,2

    Trả về array shape (T, 4).
    """
    # Lấy left foot: cột 1–8 trong DataFrame (0-indexed sau khi bỏ header)
    left = df.iloc[:, 1:9].values.astype(np.float32)  # shape (T, 8)

    S1 = (left[:, 2] + left[:, 3]) / 2.0   # metatarsal 1,2
    S2 = (left[:, 6] + left[:, 7]) / 2.0   # toe 1,2
    S3 = left[:, 5]                          # metatarsal 5
    S4 = (left[:, 0] + left[:, 1]) / 2.0   # heel 1,2

    return np.stack([S1, S2, S3, S4], axis=1)  # (T, 4)


# ---------------------------------------------------------------------------
# Bước 3: Normalize per-subject → [0, 1]
# ---------------------------------------------------------------------------

def normalize_per_subject(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize từng cột về [0, 1] dựa trên min/max của chính subject đó.
    eps tránh chia cho 0 khi sensor bị lỗi (toàn 0).
    """
    col_min = data.min(axis=0)
    col_max = data.max(axis=0)
    return (data - col_min) / (col_max - col_min + eps)


# ---------------------------------------------------------------------------
# Bước 4: Downsample 100Hz → 50Hz
# ---------------------------------------------------------------------------

def downsample_to_target(data: np.ndarray, original_hz: float, target_hz: float) -> np.ndarray:
    """
    Giảm sample rate bằng cách lấy mỗi N sample (integer decimation).
    GaitPDB: 100Hz → 50Hz  ⟹  lấy mỗi 2 sample.
    Nếu tỉ lệ không nguyên, dùng scipy.signal.decimate.
    """
    factor = original_hz / target_hz
    if not factor.is_integer():
        raise ValueError(
            f"Tỉ lệ {original_hz}/{target_hz} = {factor} không nguyên — "
            "cần dùng polyphase resampling (scipy.signal.resample_poly)"
        )
    factor = int(factor)
    if factor == 1:
        return data
    # Lấy mỗi 'factor' sample (đã filter trước rồi nên không cần anti-alias thêm)
    return data[::factor]


# ---------------------------------------------------------------------------
# Bước 5: Sliding window → sequences
# ---------------------------------------------------------------------------

def sliding_window(data: np.ndarray, window: int, stride: int) -> np.ndarray:
    """
    Cắt time-series (T, 4) thành các cửa sổ (N, window, 4).
    Bỏ qua cửa sổ cuối nếu không đủ độ dài.
    """
    sequences = []
    T = data.shape[0]
    start = 0
    while start + window <= T:
        sequences.append(data[start : start + window])
        start += stride
    if not sequences:
        return np.empty((0, window, data.shape[1]), dtype=np.float32)
    return np.stack(sequences, axis=0)  # (N, 100, 4)


# ---------------------------------------------------------------------------
# Pipeline hoàn chỉnh cho 1 file GaitPDB
# ---------------------------------------------------------------------------

def process_gaitpdb_file(filepath: Path, label: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Đọc 1 file GaitPDB .txt, chạy toàn bộ pipeline, trả về (X, y).
    X shape: (N, 100, 4)
    y shape: (N,) — toàn bộ giá trị = label
    """
    try:
        df = pd.read_csv(filepath, sep="\t", header=None, engine="python")
        if df.shape[1] < 18:
            log.warning("Bỏ qua %s — không đủ cột (%d)", filepath.name, df.shape[1])
            return np.empty((0, WINDOW_SIZE, 4)), np.empty(0)

        # Loại bỏ dòng NaN
        df = df.dropna()
        if len(df) < WINDOW_SIZE:
            log.warning("Bỏ qua %s — quá ít dòng (%d)", filepath.name, len(df))
            return np.empty((0, WINDOW_SIZE, 4)), np.empty(0)

        # Aggregate → filter → normalize → downsample → window
        data = aggregate_gaitpdb_columns(df)                                    # (T, 4) Newton
        data = apply_filter_to_array(data, LOWPASS_CUTOFF, ORIGINAL_HZ)         # filter 20Hz
        data = normalize_per_subject(data)                                       # [0, 1]
        data = downsample_to_target(data, ORIGINAL_HZ, TARGET_HZ)               # 50Hz
        X = sliding_window(data, WINDOW_SIZE, STRIDE)                            # (N, 100, 4)

        if X.shape[0] == 0:
            return X, np.empty(0)

        y = np.full(X.shape[0], label, dtype=np.int8)
        return X.astype(np.float32), y

    except Exception as e:
        log.error("Lỗi khi xử lý %s: %s", filepath.name, e)
        return np.empty((0, WINDOW_SIZE, 4)), np.empty(0)


# ---------------------------------------------------------------------------
# Pipeline cho GaitNDD (stride-to-stride format — khác GaitPDB)
# ---------------------------------------------------------------------------

def process_gaitndd_file(filepath: Path, label: int) -> tuple[np.ndarray, np.ndarray]:
    """
    GaitNDD lưu stride-interval (không phải raw force).
    Mỗi dòng = 1 stride: [time_L, time_R, swing_L, swing_R, stance_L, stance_R, ...]

    Vì không có raw pressure, ta tạo synthetic 4-channel signal từ các đặc trưng
    stride (stride_time, swing_time, stance_time, double_support) để LSTM học
    temporal pattern của gait disorder.

    Trả về (X, y) cùng shape convention với GaitPDB.
    """
    try:
        df = pd.read_csv(filepath, sep=r"\s+", header=None, engine="python")
        df = df.dropna()

        if df.shape[1] < 4 or len(df) < WINDOW_SIZE:
            log.warning("Bỏ qua %s — %d cột, %d dòng", filepath.name, df.shape[1], len(df))
            return np.empty((0, WINDOW_SIZE, 4)), np.empty(0)

        # Lấy 4 cột đầu tiên (đại diện cho 4 kênh tương đương sensor)
        data = df.iloc[:, :4].values.astype(np.float32)

        # Filter (giả sử 50Hz — NDD không ghi rõ, dùng TARGET_HZ)
        data = apply_filter_to_array(data, LOWPASS_CUTOFF, TARGET_HZ)
        data = normalize_per_subject(data)
        X = sliding_window(data, WINDOW_SIZE, STRIDE)

        if X.shape[0] == 0:
            return X, np.empty(0)

        y = np.full(X.shape[0], label, dtype=np.int8)
        return X.astype(np.float32), y

    except Exception as e:
        log.error("Lỗi khi xử lý %s: %s", filepath.name, e)
        return np.empty((0, WINDOW_SIZE, 4)), np.empty(0)


# ---------------------------------------------------------------------------
# Load toàn bộ GaitPDB
# ---------------------------------------------------------------------------

def load_gaitpdb(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    GaitPDB: tên file dạng GaXXXX.txt (Parkinson) và CoXXXX.txt (Control).
    Trả về (X, y) gộp toàn bộ subject.
    """
    if not data_dir.exists():
        log.error("Không tìm thấy thư mục GaitPDB: %s", data_dir)
        return np.empty((0, WINDOW_SIZE, 4)), np.empty(0)

    all_X, all_y = [], []

    pd_files = sorted(data_dir.glob("*Pt*.txt"))
    hc_files = sorted(data_dir.glob("*Co*.txt"))

    log.info("GaitPDB — tìm thấy %d PD files, %d Control files", len(pd_files), len(hc_files))

    for f in pd_files:
        X, y = process_gaitpdb_file(f, label=1)   # 1 = Parkinson
        if X.shape[0] > 0:
            all_X.append(X)
            all_y.append(y)

    for f in hc_files:
        X, y = process_gaitpdb_file(f, label=0)   # 0 = Normal
        if X.shape[0] > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        log.warning("GaitPDB: không load được file nào")
        return np.empty((0, WINDOW_SIZE, 4)), np.empty(0)

    return np.concatenate(all_X), np.concatenate(all_y)


# ---------------------------------------------------------------------------
# Load toàn bộ GaitNDD
# ---------------------------------------------------------------------------

def load_gaitndd(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    GaitNDD: tên file dạng park*.txt, hunt*.txt, als*.txt, control*.txt
    Huntington và ALS → label 2 (Abnormal gait)
    Parkinson → label 1, Control → label 0
    """
    if not data_dir.exists():
        log.error("Không tìm thấy thư mục GaitNDD: %s", data_dir)
        return np.empty((0, WINDOW_SIZE, 4)), np.empty(0)

    file_label_map = {
        "park":    1,   # Parkinson
        "hunt":    2,   # Abnormal
        "als":     2,   # Abnormal
        "control": 0,   # Normal
    }

    all_X, all_y = [], []

    for prefix, label in file_label_map.items():
        files = sorted(data_dir.glob(f"{prefix}*.txt"))
        log.info("GaitNDD — %s: %d files (label=%d)", prefix, len(files), label)
        for f in files:
            X, y = process_gaitndd_file(f, label)
            if X.shape[0] > 0:
                all_X.append(X)
                all_y.append(y)

    if not all_X:
        log.warning("GaitNDD: không load được file nào")
        return np.empty((0, WINDOW_SIZE, 4)), np.empty(0)

    return np.concatenate(all_X), np.concatenate(all_y)


# ---------------------------------------------------------------------------
# Xử lý data thực tế từ phần cứng (runtime inference)
# ---------------------------------------------------------------------------

def process_realtime_window(raw_adc: np.ndarray) -> np.ndarray:
    """
    Xử lý 1 cửa sổ dữ liệu thực tế từ ESP32/ADS1115 để đưa vào LSTM.

    Input:
        raw_adc : array shape (100, 4), dtype float32
                  Giá trị ADC thô: S1,S2 từ ADS1115 (0–32767),
                                   S3,S4 từ ESP32 ADC (0–4095)
    Output:
        array shape (1, 100, 4), dtype float32 — sẵn sàng cho model.predict()

    Lưu ý: normalize per-window (không per-subject) vì chỉ có 1 cửa sổ.
    Điều này có thể gây sai lệch nhỏ so với training — acceptable cho inference.
    Nếu cần chính xác hơn, lưu running min/max từ đầu session và normalize theo đó.
    """
    if raw_adc.shape != (WINDOW_SIZE, 4):
        raise ValueError(
            f"Kích thước không đúng: nhận {raw_adc.shape}, cần ({WINDOW_SIZE}, 4)"
        )

    data = raw_adc.astype(np.float32)

    # Normalize từng sensor về [0, 1] dựa trên range cứng (không dùng min/max window
    # vì window ngắn có thể không cover full range)
    data[:, 0] /= 4095.0    # S1 — ESP32 ADC 12-bit (GPIO4)
    data[:, 1] /= 26400.0   # S2 — ADS1115 GAIN_ONE (±4.096V, VCC=3.3V → max ~26400)

    data[:, 2] /= 4095.0    # S3 — ESP32 ADC 12-bit
    data[:, 3] /= 4095.0    # S4 — ESP32 ADC 12-bit

    # Low-pass filter (data thực tế đã 50Hz, không cần downsample)
    data = apply_filter_to_array(data, LOWPASS_CUTOFF, TARGET_HZ)

    # Clip để tránh giá trị ngoài [0, 1] sau filter (filtfilt có thể overshoot nhẹ)
    data = np.clip(data, 0.0, 1.0)

    return data[np.newaxis, :, :]   # (1, 100, 4)


# ---------------------------------------------------------------------------
# Split và lưu
# ---------------------------------------------------------------------------

def split_and_save(X: np.ndarray, y: np.ndarray, output_dir: Path) -> None:
    """
    Stratified split 70/15/15, lưu numpy arrays ra disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stratified split giữ tỉ lệ class
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=y,
        random_state=RANDOM_SEED,
    )
    val_fraction = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=(1 - val_fraction),
        stratify=y_tmp,
        random_state=RANDOM_SEED,
    )

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }

    for name, arr in splits.items():
        path = output_dir / f"{name}.npy"
        np.save(path, arr)
        log.info("Saved %s — shape %s", path.name, arr.shape)

    # Phân bố class
    for split_name in ["train", "val", "test"]:
        y_arr = splits[f"y_{split_name}"]
        unique, counts = np.unique(y_arr, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        log.info("%-5s class distribution: %s", split_name, dist)

    # Lưu label encoder
    le = LabelEncoder()
    le.fit(y)
    le_path = Path("ml/models") / "label_encoder.pkl"
    le_path.parent.mkdir(parents=True, exist_ok=True)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    log.info("Saved %s — classes: %s", le_path, le.classes_)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Smart Insole Preprocessing Pipeline ===")
    log.info("Window: %d steps | Stride: %d | Target: %dHz | Lowpass: %.0fHz",
             WINDOW_SIZE, STRIDE, TARGET_HZ, LOWPASS_CUTOFF)

    # Load GaitPDB
    log.info("--- Load GaitPDB ---")
    X_pdb, y_pdb = load_gaitpdb(RAW_GAITPDB)
    log.info("GaitPDB: %d windows, shape %s", X_pdb.shape[0], X_pdb.shape)

    # Load GaitNDD
    log.info("--- Load GaitNDD ---")
    X_ndd, y_ndd = load_gaitndd(RAW_GAITNDD)
    log.info("GaitNDD: %d windows, shape %s", X_ndd.shape[0], X_ndd.shape)

    # Gộp
    arrays_X, arrays_y = [], []
    if X_pdb.shape[0] > 0:
        arrays_X.append(X_pdb)
        arrays_y.append(y_pdb)
    if X_ndd.shape[0] > 0:
        arrays_X.append(X_ndd)
        arrays_y.append(y_ndd)

    if not arrays_X:
        log.error("Không có data để xử lý. Kiểm tra đường dẫn RAW_GAITPDB và RAW_GAITNDD.")
        return

    X = np.concatenate(arrays_X).astype(np.float32)
    y = np.concatenate(arrays_y).astype(np.int8)

    log.info("--- Tổng hợp ---")
    log.info("Total: %d windows | Shape X: %s | Shape y: %s", X.shape[0], X.shape, y.shape)
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        label_name = {0: "Normal", 1: "Parkinson", 2: "Abnormal"}[int(u)]
        log.info("  Label %d (%s): %d windows (%.1f%%)", u, label_name, c, 100*c/len(y))

    # Split và lưu
    log.info("--- Split và lưu ---")
    split_and_save(X, y, PROCESSED_DIR)

    log.info("=== Preprocessing hoàn tất ===")
    log.info("Output: %s", PROCESSED_DIR.resolve())


if __name__ == "__main__":
    main()