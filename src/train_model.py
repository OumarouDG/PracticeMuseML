import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocess import preprocess

DATA_PATH = "data/combined_data.csv"
MODEL_PATH = "models/trained_model.pkl"

def main():
    # Load raw EEG data
    df = pd.read_csv(DATA_PATH)

    # ===== ADD THIS SECTION HERE =====
    print("=== Training Data Analysis ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    if 'attempt_id' in df.columns:
        window_sizes = df.groupby('attempt_id').size()
        print(f"\nWindow sizes per attempt:")
        print(f"  Min: {window_sizes.min()}")
        print(f"  Max: {window_sizes.max()}")
        print(f"  Mean: {window_sizes.mean():.0f}")
        print(f"  Total attempts: {len(window_sizes)}")
    else:
        print("No 'attempt_id' column found - cannot determine window sizes")
    print("=" * 40)
    # ===== END OF ADDED SECTION =====

    # Preprocess: do not aggregate so we preserve sequence info
    processed = preprocess(df, aggregate=True, add_features=True)

    # Features and labels
    feature_cols = [c for c in processed.columns if c not in ("attempt_id", "label")]
    if "label" not in processed.columns:
        raise ValueError("No 'label' column found in the dataset.")

    X = processed[feature_cols]
    y = processed["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc:.2f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

if __name__ == "__main__":
    main()