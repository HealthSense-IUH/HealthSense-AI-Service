import numpy as np

def apply_physiological_filter(rr_intervals: list[float]) -> np.ndarray:
    rr_arr = np.array(rr_intervals)
    valid_rr = rr_arr[(rr_arr >= 300) & (rr_arr <= 2000)]
    return valid_rr
