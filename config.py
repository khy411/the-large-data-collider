from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

TRAINING_DIR = BASE_DIR / "training"
VALIDATION_DIR = BASE_DIR / "validation"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output" / "plots"