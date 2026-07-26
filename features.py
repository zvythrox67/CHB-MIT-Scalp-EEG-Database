import numpy as np
from scipy.signal import welch
import antropy as ant

_trapz = getattr(np, "trapezoid", None) or np.trapz

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

def _band_power(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return _trapz(psd[mask], freqs[mask])

def _hjorth_params(sig):
    first_deriv = np.diff(sig)
    second_deriv = np.diff(first_deriv)
    
    activity = np.var(sig)
    mobility = np.sqrt(np.var(first_deriv) / activity) if activity > 0 else 0.0
    
    var_first = np.var(first_deriv)
    mobility_of_deriv = np.sqrt(np.var(second_deriv) / var_first) if var_first > 0 else 0.0
    complexity = mobility_of_deriv / mobility if mobility > 0 else 0.0
    
    return activity, mobility, complexity

def _channel_features(sig, sfreq):
    feats = {}
    
    feats["mean"] = np.mean(sig)
    feats["std"] = np.std(sig)
    feats["rms"] = np.sqrt(np.mean(sig ** 2))
    feats["ptp"] = np.ptp(sig)
    feats["line_length"] = np.sum(np.abs(np.diff(sig)))