import os
import joblib
import numpy as np
from sklearn.cluster import KMeans
from utils import load_eeg_data
from preprocess import preprocess

DATA_PATH = "data/raw/sample_data.csv"
MODEL_PATH = "models/trained_model.pkl"

def main():
    # Load and preprocess raw EEG data
    df = load_eeg_data(DATA_PATH)
    processed = preprocess(df)

    # Use core EEG features
    X = processed[["alpha", "beta", "theta", "gamma"]]

    # Unsupervised clustering (4 clusters → up, down, left, right)
    model = KMeans(n_clusters=4, random_state=42)
    model.fit(X)

    # Print cluster summary
    labels, counts = np.unique(model.labels_, return_counts=True)
    print("Cluster distribution:", dict(zip(labels, counts)))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved unsupervised model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
