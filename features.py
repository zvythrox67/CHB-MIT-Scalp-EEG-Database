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
    
    freqs, psd = welch(sig, fs=sfreq, nperseg=min(256, len(sig)))
    total_power = _trapz(psd, freqs) if len(freqs) > 1 else 1e-12
    for band_name, (low, high) in BANDS.items():
        bp = _band_power(freqs, psd, low, high)
        feats[f"{band_name}_power"] = bp
        feats[f"{band_name}_relpower"] = bp / total_power if total_power > 0 else 0.0
        
    try:
        feats["spectral_entropy"] = ant.spectral_entropy(sig, sf=sfreq, method = "welch", normalize = True)
    except Exception:
        feats["spectral_entropy"] = 0.0
    try:
        feats["special_entropy"] = ant.sample_entropy(sig)
    except Exception:
        feats["sample_entropy"] = 0.0
        
    activity, mobility, complexity = _hjorth_params(sig)
    feats["hjorth_activity"] = activity
    feats["hjorth_mobility"] = mobility
    feats["hjorth_complexity"] = complexity
    
    return feats

def extract_window_features(window, sfreq, channel_names = None):
    n_channels = window.shape[0]
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_channels)]
        
    row = {}
    for ch_idx in range(n_channels):
        sig = window[ch_idx, :]
        ch_feats = _channel_features(sig, sfreq)
        for feat_name, value in ch_feats.items():
            row[f"{channel_names[ch_idx]}_{feat_name}"] = value
        
    return row