import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch

def bandpass_filter(data, lowcut=1.0, highcut=40.0, fs=256, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, freq=60.0, fs=256, quality=30):
    from scipy.signal import iirnotch
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data)

def remove_blink_artifacts(df, threshold=150.0):
    """Interpolate spikes above threshold to clean blink/muscle artifacts."""
    features = ["alpha", "beta", "theta", "gamma"]
    for f in features:
        if f in df.columns:
            mask = df[f].abs() > threshold
            df.loc[mask, f] = np.nan
            df[f] = df[f].interpolate().bfill().ffill()
    return df

def add_psd_features(df, fs=256):
    """Adds average power spectral density (PSD) features for each EEG band."""
    bands = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
    psd_features = []
    for f in ["alpha", "beta", "theta", "gamma"]:
        if f not in df.columns:
            continue
        data = df[f].values
        freqs, psd = welch(data, fs=fs, nperseg=256)
        for band, (low, high) in bands.items():
            band_power = psd[(freqs >= low) & (freqs <= high)].mean()
            df[f"{f}_{band}_power"] = band_power
            psd_features.append(f"{f}_{band}_power")
    return df, psd_features

def add_interaction_features(df):
    """Add interaction (cross and self) power features between EEG bands."""
    features = ["alpha", "beta", "theta", "gamma"]
    for a in features:
        for b in features:  # include self-pairs
            if a in df.columns and b in df.columns:
                df[f"{a}_{b}_power"] = df[a] * df[b]
    return df

def preprocess(df: pd.DataFrame, aggregate: bool = True, add_features: bool = True) -> pd.DataFrame:
    features = ["alpha", "beta", "theta", "gamma"]

    # Apply bandpass and notch filters
    for f in features:
        if f in df.columns:
            df[f] = bandpass_filter(df[f].values)
            df[f] = notch_filter(df[f].values)

    # Remove artifacts
    df = remove_blink_artifacts(df)

    # Add PSD and interaction features
    psd_features = []
    if add_features:
        df, psd_features = add_psd_features(df)
        df = add_interaction_features(df)

    if aggregate:
        group_cols = []
        if "attempt_id" in df.columns:
            group_cols.append("attempt_id")
        if "label" in df.columns:
            group_cols.append("label")

        if group_cols:
            aggregated = df.groupby(group_cols)[features + psd_features].mean().reset_index()
        else:
            aggregated = df[features + psd_features].mean().to_frame().T

        return aggregated
    else:
        return df
