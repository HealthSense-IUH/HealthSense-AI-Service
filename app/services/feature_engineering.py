"""Service trích xuất 16 đặc trưng HRV từ chuỗi R-R Interval.

Chuyển đổi từ Notebook 02 (02_feature_engineering.ipynb) thành Python module.

Tham khảo:
    [1] Task Force of ESC/NASPE, European Heart Journal, vol. 17, pp. 354-381, 1996.
    [2] Malik et al., European Heart Journal, 1989.
    [3] Shaffer & Ginsberg, Frontiers in Public Health, vol. 5, 2017.
"""

import numpy as np
from scipy.signal import welch
from scipy import interpolate


def calculate_rr_intervals(
    peak_indices: np.ndarray,
    timestamps: np.ndarray,
    min_rr: float = 300.0,
    max_rr: float = 2000.0,
) -> np.ndarray:
    """Tính R-R Intervals từ các đỉnh nhịp tim và lọc theo tiêu chí Malik [2].

    Parameters:
        peak_indices: Mảng chỉ số của các đỉnh
        timestamps: Mảng timestamp tương ứng (ms)
        min_rr: R-R tối thiểu cho phép (ms) - tương đương 200 BPM
        max_rr: R-R tối đa cho phép (ms) - tương đương 30 BPM

    Returns:
        Mảng R-R Intervals hợp lệ (ms)
    """
    if len(peak_indices) < 2:
        return np.array([])

    peak_times = timestamps[peak_indices]
    rr_intervals = np.diff(peak_times)

    # Lọc theo tiêu chí Malik
    rr_valid = rr_intervals[(rr_intervals >= min_rr) & (rr_intervals <= max_rr)]

    return rr_valid


def calculate_time_domain(rr_intervals: np.ndarray) -> dict:
    """Trích xuất 6 đặc trưng miền thời gian [1].

    Parameters:
        rr_intervals: Mảng R-R Intervals (ms)

    Returns:
        dict chứa Mean_NN, SDNN, RMSSD, pNN50, NN50, CV
    """
    nan_result = {
        "mean_nn": None, "sdnn": None, "rmssd": None,
        "pnn50": None, "nn50": None, "cv": None,
    }

    if len(rr_intervals) < 2:
        return nan_result

    mean_nn = float(np.mean(rr_intervals))
    sdnn = float(np.std(rr_intervals, ddof=1))

    diff_nn = np.diff(rr_intervals)
    rmssd = float(np.sqrt(np.mean(diff_nn**2)))

    nn50 = int(np.sum(np.abs(diff_nn) > 50))
    pnn50 = float((nn50 / len(diff_nn)) * 100)

    cv = float((sdnn / mean_nn) * 100) if mean_nn > 0 else None

    return {
        "mean_nn": round(mean_nn, 2),
        "sdnn": round(sdnn, 2),
        "rmssd": round(rmssd, 2),
        "pnn50": round(pnn50, 2),
        "nn50": nn50,
        "cv": round(cv, 2) if cv is not None else None,
    }


def calculate_frequency_domain(
    rr_intervals: np.ndarray, fs_interp: float = 4.0
) -> dict:
    """Trích xuất 6 đặc trưng miền tần số bằng Welch's Periodogram [1].

    Parameters:
        rr_intervals: Mảng R-R Intervals (ms)
        fs_interp: Tần số nội suy (Hz)

    Returns:
        dict chứa LF, HF, LF_HF_Ratio, LF_norm, HF_norm, Total_Power
    """
    nan_result = {
        "lf": None, "hf": None, "lf_hf_ratio": None,
        "lf_norm": None, "hf_norm": None, "total_power": None,
    }

    if len(rr_intervals) < 10:
        return nan_result

    # Trục thời gian tích lũy (giây)
    t = np.cumsum(rr_intervals) / 1000.0
    t -= t[0]

    # Nội suy Cubic Spline
    t_interp = np.arange(0, t[-1], 1.0 / fs_interp)
    f_interp = interpolate.interp1d(
        t, rr_intervals, kind="cubic", fill_value="extrapolate"
    )
    rr_interp = f_interp(t_interp)

    # Detrend
    rr_interp -= np.mean(rr_interp)

    # Welch's Periodogram
    f, pxx = welch(rr_interp, fs=fs_interp, nperseg=min(len(rr_interp), 256))

    # Năng lượng phổ theo dải tần
    lf_mask = (f >= 0.04) & (f < 0.15)
    hf_mask = (f >= 0.15) & (f < 0.40)

    lf = float(np.trapezoid(pxx[lf_mask], f[lf_mask])) if np.any(lf_mask) else None
    hf = float(np.trapezoid(pxx[hf_mask], f[hf_mask])) if np.any(hf_mask) else None

    if lf is None or hf is None:
        return nan_result

    lf_hf_ratio = lf / hf if hf > 0 else None
    total_power = lf + hf
    lf_norm = (lf / total_power) * 100 if total_power > 0 else None
    hf_norm = (hf / total_power) * 100 if total_power > 0 else None

    return {
        "lf": round(lf, 4),
        "hf": round(hf, 4),
        "lf_hf_ratio": round(lf_hf_ratio, 4) if lf_hf_ratio is not None else None,
        "lf_norm": round(lf_norm, 2) if lf_norm is not None else None,
        "hf_norm": round(hf_norm, 2) if hf_norm is not None else None,
        "total_power": round(total_power, 4),
    }


def calculate_spo2_features(spo2_values: np.ndarray) -> dict:
    """Trích xuất 3 đặc trưng từ dữ liệu SpO2.

    Parameters:
        spo2_values: Mảng giá trị SpO2

    Returns:
        dict chứa SpO2_mean, SpO2_std, SpO2_min
    """
    # Loại bỏ giá trị 0 (cảm biến chưa ổn định)
    spo2_clean = spo2_values[spo2_values > 0]

    if len(spo2_clean) < 10:
        return {"spo2_mean": None, "spo2_std": None, "spo2_min": None}

    return {
        "spo2_mean": round(float(np.mean(spo2_clean)), 2),
        "spo2_std": round(float(np.std(spo2_clean)), 2),
        "spo2_min": round(float(np.min(spo2_clean)), 2),
    }


def extract_all_features(
    rr_intervals: np.ndarray,
    spo2_values: np.ndarray,
) -> dict:
    """Trích xuất toàn bộ 16 đặc trưng HRV + SpO2.

    Parameters:
        rr_intervals: Mảng R-R Intervals (ms)
        spo2_values: Mảng giá trị SpO2

    Returns:
        dict chứa 16 features
    """
    # Time-domain (6 features)
    td = calculate_time_domain(rr_intervals)

    # HR_mean (1 feature)
    hr_mean = round(60000 / td["mean_nn"], 1) if td["mean_nn"] else None

    # Frequency-domain (6 features)
    fd = calculate_frequency_domain(rr_intervals)

    # SpO2 (3 features)
    spo2 = calculate_spo2_features(spo2_values)

    return {**td, "hr_mean": hr_mean, **fd, **spo2}
