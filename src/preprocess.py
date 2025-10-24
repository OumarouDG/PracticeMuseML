import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut=1.0, highcut=40.0, fs=256, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    features = ["alpha", "beta", "theta", "gamma"]

    # Apply bandpass filter only to columns that exist
    for f in features:
        if f in df.columns:
            df[f] = bandpass_filter(df[f].values)

    # Group by attempt (labels optional)
    group_cols = ["attempt_id"]
    if "label" in df.columns:
        group_cols.append("label")

    aggregated = df.groupby(group_cols)[features].mean().reset_index()

    # Drop label column if present (for unsupervised learning)
    if "label" in aggregated.columns:
        aggregated = aggregated.drop(columns=["label"])

    return aggregated
