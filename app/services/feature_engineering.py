import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d

def calculate_time_domain(rr_intervals: np.ndarray) -> dict:
    if len(rr_intervals) < 2:
        return {k: np.nan for k in ['Mean_NN', 'SDNN', 'RMSSD', 'pNN50', 'NN50', 'CV']}
    
    mean_nn = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1)
    
    diff_nn = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_nn**2))
    
    nn50 = np.sum(np.abs(diff_nn) > 50)
    pnn50 = (nn50 / len(diff_nn)) * 100 if len(diff_nn) > 0 else 0
    
    cv = (sdnn / mean_nn) * 100 if mean_nn > 0 else np.nan
    
    return {
        'Mean_NN': float(mean_nn),
        'SDNN': float(sdnn),
        'RMSSD': float(rmssd),
        'pNN50': float(pnn50),
        'NN50': float(nn50),
        'CV': float(cv)
    }

def calculate_frequency_domain(rr_intervals: np.ndarray, fs_interp=4.0) -> dict:
    nan_result = {k: np.nan for k in ['LF', 'HF', 'LF_HF_Ratio', 'LF_norm', 'HF_norm', 'Total_Power']}
    if len(rr_intervals) < 10:
        return nan_result
    
    try:
        t = np.cumsum(rr_intervals) / 1000.0
        t -= t[0]
        
        rr_sec = rr_intervals / 1000.0
        f_interp = interp1d(t, rr_sec, kind='cubic')
        t_interp = np.arange(t[0], t[-1], 1.0 / fs_interp)
        rr_interp = f_interp(t_interp)
        
        nperseg = min(256, len(rr_interp))
        if nperseg < 2:
            return nan_result
            
        f, pxx = welch(rr_interp, fs=fs_interp, nperseg=nperseg)
        
        lf_band = (f >= 0.04) & (f <= 0.15)
        hf_band = (f >= 0.15) & (f <= 0.4)
        
        df = f[1] - f[0]
        lf = np.sum(pxx[lf_band]) * df
        hf = np.sum(pxx[hf_band]) * df
        
        total_power = lf + hf
        lf_hf_ratio = lf / hf if hf > 0 else np.nan
        
        lf_norm = (lf / total_power) * 100 if total_power > 0 else np.nan
        hf_norm = (hf / total_power) * 100 if total_power > 0 else np.nan
        
        return {
            'LF': float(lf),
            'HF': float(hf),
            'LF_HF_Ratio': float(lf_hf_ratio),
            'LF_norm': float(lf_norm),
            'HF_norm': float(hf_norm),
            'Total_Power': float(total_power)
        }
    except Exception:
        return nan_result

def extract_hrv_features(valid_rr: np.ndarray) -> dict:
    if len(valid_rr) < 10:
        raise ValueError("Không đủ nhịp tim để trích xuất đặc trưng (Yêu cầu >= 10).")

    features = {
        'HR_mean': np.nan,
        'Mean_NN': np.nan, 'SDNN': np.nan, 'RMSSD': np.nan, 
        'pNN50': np.nan, 'NN50': np.nan, 'CV': np.nan,
        'LF': np.nan, 'HF': np.nan, 'LF_HF_Ratio': np.nan,
        'LF_norm': np.nan, 'HF_norm': np.nan, 'Total_Power': np.nan
    }
    
    features['HR_mean'] = float(60000.0 / np.mean(valid_rr))
    features.update(calculate_time_domain(valid_rr))
    features.update(calculate_frequency_domain(valid_rr))
    
    # Fill NaN with 0 for safety before feeding to the model
    for k, v in features.items():
        if np.isnan(v):
            features[k] = 0.0
            
    return features
