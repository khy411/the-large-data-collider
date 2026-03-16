import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

# master orchestrator, runs the full pipeline end to end
# usage:
#   python pipeline.py (pandas sample mode, 50k records)
#   python pipeline.py --full (pandas full mode, all records)
#   python pipeline.py --spark (pyspark distributed mode, all 11m records)

import argparse

def run_pandas_pipeline(full=False):
    from src.parser import tfrecords_to_dataframe
    from src.cleaner import validate_and_clean
    from src.writer import save_parquet
    from src.analytics import (
        plot_feature_distributions,
        plot_correlation_heatmap,
        plot_auc_separation,
        plot_missing_jet_rate,
        plot_invariant_mass,
        print_summary_stats
    )
    from config import TRAINING_DIR

    train_files = sorted(TRAINING_DIR.glob("*.tfrecord"))
    max_records = None if full else 50000

    print("Parsing...")
    df = tfrecords_to_dataframe(train_files, max_records=max_records)

    print("Cleaning...")
    df_clean = validate_and_clean(df)

    print("Writing parquet...")
    save_parquet(df_clean, "higgs_train_sample.parquet")

    print("Generating analytics...")
    print_summary_stats(df_clean)
    plot_feature_distributions(df_clean)
    plot_correlation_heatmap(df_clean)
    plot_auc_separation(df_clean)
    plot_missing_jet_rate(df_clean)
    plot_invariant_mass(df_clean)

    print("\nPipeline complete")

def run_spark_pipeline():
    from src.spark_pipeline import run_spark_pipeline
    run_spark_pipeline(mode="full")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Large Data Collider Pipeline")
    parser.add_argument("--full", action="store_true", help="Run pandas pipeline on all records")
    parser.add_argument("--spark", action="store_true", help="Run distributed pyspark pipeline")
    args = parser.parse_args()

    if args.spark:
        print("Running spark pipeline...")
        run_spark_pipeline()
    elif args.full:
        print("Running full pandas pipeline...")
        run_pandas_pipeline(full=True)
    else:
        print("Running sample pandas pipeline (50k records)...")
        run_pandas_pipeline(full=False)