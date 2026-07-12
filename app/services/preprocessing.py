"""Service tiền xử lý tín hiệu PPG.

Chuyển đổi từ Notebook 01 (01_preprocessing.ipynb) thành Python module.
Áp dụng Butterworth Bandpass Filter để loại bỏ Baseline Wander và nhiễu cao tần.

Tham khảo:
    [1] Task Force of ESC/NASPE, European Heart Journal, vol. 17, pp. 354-381, 1996.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Áp dụng bộ lọc Butterworth Bandpass Filter.

    Parameters:
        data: Tín hiệu đầu vào
        lowcut: Tần số cắt dưới (Hz)
        highcut: Tần số cắt trên (Hz)
        fs: Tần số lấy mẫu (Hz)
        order: Bậc bộ lọc

    Returns:
        Tín hiệu đã lọc
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data)
    return y


def preprocess_signal(
    ir_data: np.ndarray,
    sampling_rate: float = 100.0,
    bandpass_low: float = 0.7,
    bandpass_high: float = 4.0,
    filter_order: int = 4,
) -> np.ndarray:
    """Tiền xử lý tín hiệu IR từ cảm biến MAX30102.

    Quy trình:
        1. Zero-centering (trừ giá trị trung bình)
        2. Bandpass Filter [0.7 - 4.0] Hz

    Parameters:
        ir_data: Mảng giá trị IR thô
        sampling_rate: Tần số lấy mẫu (Hz)
        bandpass_low: Tần số cắt dưới (Hz)
        bandpass_high: Tần số cắt trên (Hz)
        filter_order: Bậc bộ lọc Butterworth

    Returns:
        Tín hiệu IR đã lọc (zero-centered)
    """
    # Zero-centering
    ir_centered = ir_data - np.mean(ir_data)

    # Bandpass Filter
    ir_filtered = butter_bandpass_filter(
        ir_centered, bandpass_low, bandpass_high, sampling_rate, filter_order
    )

    return ir_filtered


def detect_peaks(
    filtered_signal: np.ndarray,
    sampling_rate: float = 100.0,
    min_bpm: float = 30.0,
    max_bpm: float = 200.0,
    prominence: float = 10.0,
) -> np.ndarray:
    """Phát hiện đỉnh nhịp tim (Systolic Peaks) trên tín hiệu đã lọc.

    Parameters:
        filtered_signal: Tín hiệu IR đã qua Bandpass Filter
        sampling_rate: Tần số lấy mẫu (Hz)
        min_bpm: Nhịp tim tối thiểu cho phép (BPM)
        max_bpm: Nhịp tim tối đa cho phép (BPM)
        prominence: Ngưỡng prominence tối thiểu để loại đỉnh giả

    Returns:
        Mảng chỉ số (index) của các đỉnh nhịp tim
    """
    min_distance = int((60.0 / max_bpm) * sampling_rate)

    peaks, _ = find_peaks(
        filtered_signal, distance=min_distance, prominence=prominence
    )

    return peaks
