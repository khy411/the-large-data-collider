from pathlib import Path

# Base project directory — automatically resolves to wherever the repo is cloned
BASE_DIR = Path(__file__).resolve().parent

# Data paths
TRAINING_DIR = BASE_DIR / "training"
VALIDATION_DIR = BASE_DIR / "validation"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output" / "plots"