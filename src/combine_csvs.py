import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split

# Configuration
DATA_FOLDER = "data/raw"
COMBINED_FILE = "data/combined_data.csv"
TRAIN_FILE = "data/train_data.csv"
VAL_FILE = "data/val_data.csv"
LINES_PER_ATTEMPT = 1129
VAL_RATIO = 0.2  # 20% of attempts for validation

def combine_csvs_fixed_attempts(folder_path, lines_per_attempt):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in folder: {folder_path}")

    combined_df = pd.DataFrame()
    current_attempt_id = 0

    for file in csv_files:
        df = pd.read_csv(file)

        n_rows = len(df)
        n_attempts = n_rows // lines_per_attempt
        remainder = n_rows % lines_per_attempt

        attempt_ids = []
        for i in range(n_attempts):
            attempt_ids += [current_attempt_id + i] * lines_per_attempt
        if remainder > 0:
            attempt_ids += [current_attempt_id + n_attempts] * remainder

        df["attempt_id"] = attempt_ids
        current_attempt_id = df["attempt_id"].max() + 1

        combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

def split_train_val(df, val_ratio):
    # Split at attempt level
    unique_attempts = df["attempt_id"].unique()
    train_attempts, val_attempts = train_test_split(unique_attempts, test_size=val_ratio, random_state=42)

    train_df = df[df["attempt_id"].isin(train_attempts)].reset_index(drop=True)
    val_df = df[df["attempt_id"].isin(val_attempts)].reset_index(drop=True)

    # Drop attempt_id and label from validation CSV
    val_df_no_meta = val_df.drop(columns=[c for c in ["attempt_id", "label"] if c in val_df.columns])

    return train_df, val_df_no_meta

if __name__ == "__main__":
    combined_df = combine_csvs_fixed_attempts(DATA_FOLDER, LINES_PER_ATTEMPT)
    combined_df.to_csv(COMBINED_FILE, index=False)
    print(f"Saved combined CSV: {COMBINED_FILE}, total attempts: {combined_df['attempt_id'].nunique()}")

    train_df, val_df = split_train_val(combined_df, VAL_RATIO)
    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)
    print(f"Saved training CSV: {TRAIN_FILE}, validation CSV (no attempt_id/label): {VAL_FILE}")
    print(f"Training attempts: {train_df['attempt_id'].nunique()}, Validation attempts: {val_df.shape[0] // LINES_PER_ATTEMPT}")
