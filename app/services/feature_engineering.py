import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

def extract_hrv_features(valid_rr: np.ndarray) -> dict:
    if len(valid_rr) < 10:
        raise ValueError("Không đủ nhịp tim để trích xuất đặc trưng (Yêu cầu >= 10).")

    # 1. Time-domain
    rr_diff = np.diff(valid_rr)
    mean_nni = np.mean(valid_rr)
    sdnn = np.std(valid_rr, ddof=1)
    rmssd = np.sqrt(np.mean(rr_diff**2))
    pnn50 = (np.sum(np.abs(rr_diff) > 50) / len(rr_diff)) * 100
    cvnni = sdnn / mean_nni if mean_nni > 0 else 0
    cvsd = rmssd / mean_nni if mean_nni > 0 else 0
    
    # 2. Frequency-domain
    try:
        time_arr = np.cumsum(valid_rr) / 1000.0
        fs_interp = 4.0
        time_interp = np.arange(time_arr[0], time_arr[-1], 1/fs_interp)
        f_interp = interp1d(time_arr, valid_rr, kind='cubic', fill_value="extrapolate")
        rr_interp = f_interp(time_interp)
        nperseg = min(len(rr_interp), 256)
        f, pxx = signal.welch(rr_interp, fs=fs_interp, nperseg=nperseg)
        lf_band = (f >= 0.04) & (f < 0.15)
        hf_band = (f >= 0.15) & (f < 0.4)
        lf = np.trapz(pxx[lf_band], f[lf_band]) if np.sum(lf_band) > 0 else 0
        hf = np.trapz(pxx[hf_band], f[hf_band]) if np.sum(hf_band) > 0 else 0
        lf_hf_ratio = lf / hf if hf > 0 else 0
        total_power = np.trapz(pxx, f)
    except Exception:
        lf = hf = lf_hf_ratio = total_power = 0
        
    return {
        'mean_nni': float(mean_nni),
        'sdnn': float(sdnn),
        'rmssd': float(rmssd),
        'pnn50': float(pnn50),
        'cvnni': float(cvnni),
        'cvsd': float(cvsd),
        'lf': float(lf),
        'hf': float(hf),
        'lf_hf_ratio': float(lf_hf_ratio),
        'total_power': float(total_power)
    }
