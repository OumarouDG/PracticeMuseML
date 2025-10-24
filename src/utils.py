import pandas as pd

def load_eeg_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding='utf-8-sig')  # handles BOMs automatically
    df.columns = df.columns.str.strip()  # removes leading/trailing spaces
    return df
