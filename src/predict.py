import joblib
import pandas as pd
from preprocess import preprocess

MODEL_PATH = "models/trained_model.pkl"

def predict_from_file(csv_path):
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(csv_path)
    processed = preprocess(df)
    X = processed[["alpha", "beta", "theta", "gamma"]]
    predictions = model.predict(X)
    processed["predicted_label"] = predictions
    return processed[["attempt_id", "predicted_label"]]

if __name__ == "__main__":
    result = predict_from_file("data/raw/test_data.csv")
    print(result)
