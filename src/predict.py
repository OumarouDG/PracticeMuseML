import joblib
import pandas as pd
from preprocess import preprocess

MODEL_PATH = "models/trained_model.pkl"

def predict_from_file(csv_path, window_size=256, stride=128):
    """
    Detect patterns in continuous unlabeled EEG data.
    
    Args:
        csv_path: Path to validation data
        window_size: Number of rows per window (should be >= 256 for Welch's method)
        stride: Step size for sliding window
    """
    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Read raw EEG data
    df = pd.read_csv(csv_path)
    
    print(f"Total data length: {len(df)} samples")
    print(f"Using window_size={window_size}, stride={stride}")
    
    results = []
    total_windows = (len(df) - window_size) // stride + 1
    
    # Slide window across the data
    for idx, i in enumerate(range(0, len(df) - window_size + 1, stride)):
        if idx % 10 == 0:  # Progress indicator
            print(f"Processing window {idx+1}/{total_windows}...")
        
        window_df = df.iloc[i:i+window_size].copy()
        
        # Add a temporary attempt_id for this window
        window_df['attempt_id'] = f'window_{i}'
        
        try:
            # Preprocess this window (with aggregation to get features)
            processed = preprocess(window_df, aggregate=True, add_features=True)
            
            # Get features
            feature_cols = [c for c in processed.columns if c not in ("attempt_id", "label")]
            X = processed[feature_cols]
            
            # Predict
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            results.append({
                'start_index': i,
                'end_index': i + window_size,
                'predicted_label': prediction,
                'confidence': probability.max(),
                'window_id': f'window_{i}'
            })
        except Exception as e:
            print(f"Error processing window at index {i}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    print(f"\nTotal windows processed: {len(results_df)}")
    print(f"Predictions summary:")
    print(results_df['predicted_label'].value_counts())
    
    return results_df

if __name__ == "__main__":
    # Use window_size >= 256 to avoid Welch warnings
    result = predict_from_file("data/val_data.csv", window_size=256, stride=128)
    
    print("\nAll predictions:")
    print(result)
    
    # If you want to filter by confidence
    high_conf = result[result['confidence'] > 0.7]
    print(f"\nHigh confidence predictions (>0.7): {len(high_conf)}")
    print(high_conf)
    
    # Save results
    result.to_csv("data/predictions.csv", index=False)
    print("\nSaved predictions to data/predictions.csv")