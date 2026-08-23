import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, find_peaks


def calculate_time_domain(rr_intervals: np.ndarray) -> dict:
    if len(rr_intervals) < 2:
        return dict.fromkeys(["Mean_NN", "SDNN", "RMSSD", "pNN50", "NN50", "CV"], np.nan)

    mean_nn = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)

    diff_nn = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_nn**2))

    nn50 = np.sum(np.abs(diff_nn) > 50)
    pnn50 = (nn50 / len(diff_nn)) * 100 if len(diff_nn) > 0 else 0

    cv = (sdnn / mean_nn) * 100 if mean_nn > 0 else np.nan

    return {
        "Mean_NN": float(mean_nn),
        "SDNN": float(sdnn),
        "RMSSD": float(rmssd),
        "pNN50": float(pnn50),
        "NN50": float(nn50),
        "CV": float(cv),
    }


def calculate_frequency_domain(rr_intervals: np.ndarray, fs_interp=4.0) -> dict:
    nan_result = dict.fromkeys(
        ["LF", "HF", "LF_HF_Ratio", "LF_norm", "HF_norm", "Total_Power"], 0.0
    )
    if len(rr_intervals) < 3:
        return nan_result

    try:
        rr_ms = rr_intervals
        time_rr = np.cumsum(rr_intervals) / 1000.0
        time_4hz = np.arange(time_rr[0], time_rr[-1], 0.25)

        if len(time_4hz) > 8:
            rr_4hz = np.interp(time_4hz, time_rr, rr_ms)
            mean_4hz = np.mean(rr_4hz)
            N = len(time_4hz)
            yf = (np.abs(rfft(rr_4hz - mean_4hz)) ** 2) / N
            xf = rfftfreq(N, 0.25)

            lf_band = (xf >= 0.04) & (xf < 0.15)
            hf_band = (xf >= 0.15) & (xf < 0.40)

            lf = float(np.sum(yf[lf_band])) if np.any(lf_band) else 0.0
            hf = float(np.sum(yf[hf_band])) if np.any(hf_band) else 0.0
            total_power = float(np.sum(yf))
            lf_hf_ratio = float(lf / hf) if hf > 0 else 0.0
            lf_norm = float((lf / (lf + hf + 1e-6)) * 100.0)
            hf_norm = float((hf / (lf + hf + 1e-6)) * 100.0)

            return {
                "LF": lf,
                "HF": hf,
                "LF_HF_Ratio": lf_hf_ratio,
                "LF_norm": lf_norm,
                "HF_norm": hf_norm,
                "Total_Power": total_power,
            }
        else:
            return nan_result
    except Exception:
        return nan_result


def calculate_nonlinear_domain(rr_intervals: np.ndarray, sdnn: float) -> dict:
    nan_result = dict.fromkeys(["SD1", "SD2", "SampEn"], 0.0)
    if len(rr_intervals) < 3:
        return nan_result

    try:
        diff_nn = np.diff(rr_intervals)

        sd1 = np.sqrt(0.5 * np.var(diff_nn)) if len(diff_nn) > 0 else 0.0
        sd2_val = 2 * np.var(rr_intervals) - 0.5 * np.var(diff_nn)
        sd2 = np.sqrt(max(0, sd2_val)) if len(diff_nn) > 0 else 0.0

        sampen = float(np.std(diff_nn) / (sdnn + 1e-6))

        return {"SD1": float(sd1), "SD2": float(sd2), "SampEn": float(sampen)}
    except Exception:
        return nan_result


def extract_hrv_features(valid_rr: np.ndarray) -> dict:
    if len(valid_rr) < 10:
        raise ValueError("Không đủ nhịp tim để trích xuất đặc trưng (Yêu cầu >= 10).")

    features = {
        "HR_mean": np.nan,
        "Mean_NN": np.nan,
        "SDNN": np.nan,
        "RMSSD": np.nan,
        "pNN50": np.nan,
        "NN50": np.nan,
        "CV": np.nan,
        "LF": np.nan,
        "HF": np.nan,
        "LF_HF_Ratio": np.nan,
        "LF_norm": np.nan,
        "HF_norm": np.nan,
        "Total_Power": np.nan,
        "SD1": np.nan,
        "SD2": np.nan,
        "SampEn": np.nan,
    }

    features["HR_mean"] = float(60000.0 / np.mean(valid_rr))
    time_feats = calculate_time_domain(valid_rr)
    features.update(time_feats)
    features.update(calculate_frequency_domain(valid_rr))
    features.update(calculate_nonlinear_domain(valid_rr, sdnn=time_feats.get("SDNN", 0.0)))

    # Fill NaN with 0 for safety before feeding to the model
    for k, v in features.items():
        if np.isnan(v):
            features[k] = 0.0

    return features


def butter_bandpass_filter(data, lowcut=0.5, highcut=8.0, fs=125.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band", output="ba")  # type: ignore
    y = filtfilt(b, a, data)
    return y


def extract_features_from_csv_data(time_ms_array, ppg_array, fs=125.0):
    # Lọc nhiễu (Tạm tắt để giống ML 100%)
    # clean_ppg = butter_bandpass_filter(ppg_array, fs=fs)
    clean_ppg = ppg_array

    # Tìm đỉnh giống ML workspace
    min_distance = int(0.4 * fs)
    height_thresh = np.mean(clean_ppg) + 0.3 * np.std(clean_ppg)
    peaks, _ = find_peaks(clean_ppg, distance=min_distance, height=height_thresh)

    if len(peaks) < 10:
        raise ValueError("Không tìm đủ số nhịp tim (>=10 đỉnh) trong dữ liệu.")

    # Tính R-R Intervals (ms)
    peak_times = time_ms_array[peaks]
    rr_intervals = np.diff(peak_times)

    # BỘ LỌC SINH LÝ (Tạm tắt để giống ML 100%)
    # valid_rr = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]
    valid_rr = rr_intervals

    if len(valid_rr) < 10:
        raise ValueError("Sau khi lọc, số lượng nhịp tim hợp lệ không đủ 10.")

    features = extract_hrv_features(valid_rr)
    return features
