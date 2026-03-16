import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TRAINING_DIR, PROCESSED_DIR

def save_parquet(df: pd.DataFrame, filename: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_parquet(out_path, index=False, engine="pyarrow")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {out_path}")
    print(f"Size:  {size_mb:.2f} MB")
    print(f"Shape: {df.shape}")
    return out_path

def load_parquet(filename: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / filename, engine="pyarrow")

if __name__ == "__main__":
    from parser import tfrecords_to_dataframe
    from cleaner import validate_and_clean

    train_files = sorted(TRAINING_DIR.glob("*.tfrecord"))
    print(f"Found {len(train_files)} training files")
    print("Parsing...")
    df = tfrecords_to_dataframe(train_files, max_records=50000)
    print("Cleaning...")
    df_clean = validate_and_clean(df)
    print("Writing Parquet...")
    save_parquet(df_clean, "higgs_train_sample.parquet")

    df_verify = load_parquet("higgs_train_sample.parquet")
    print(f"\nRound-trip verify — shape: {df_verify.shape}")