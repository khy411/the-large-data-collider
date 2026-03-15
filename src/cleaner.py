import pandas as pd
import numpy as np

PHYSICS_ZERO_FEATURES = [
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_b_tag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_b_tag",
]

def fix_physics_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace negative pT sentinel values with 0 (no jet detected)."""
    pt_cols = ["jet1_pt", "jet2_pt", "jet3_pt", "jet4_pt"]
    for col in pt_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"  Fixing {neg_count} negative sentinel values in {col} to 0")
            df[col] = df[col].clip(lower=0)
    return df

def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Input shape: {df.shape}")
    original_len = len(df)

    df = df.replace([np.inf, -np.inf], np.nan)

    nan_counts = df.drop(columns=PHYSICS_ZERO_FEATURES).isnull().sum()
    if nan_counts.sum() > 0:
        print(f"NaN counts:\n{nan_counts[nan_counts > 0]}")
    else:
        print("No NaNs found")

    df = df.dropna(subset=["label"])
    df = df[df["label"].isin([0.0, 1.0])]
    df["label"] = df["label"].astype(int)

    df = fix_physics_sentinels(df)

    print(f"Output shape: {df.shape}")
    print(f"Rows dropped: {original_len - len(df)}")
    return df