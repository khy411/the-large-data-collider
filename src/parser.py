import tensorflow as tf
import numpy as np
import pandas as pd
import struct
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TRAINING_DIR

FEATURE_NAMES = [
    "lepton_pT", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_b_tag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_b_tag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_b_tag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_b_tag",
    "m_jj", "m_jjj", "m_lv", "m_jlv",
    "m_bb", "m_wbb", "m_wwbb"
]

HEADER_MARKER = b'\x22\x70'

def parse_record(raw_record):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    label = example.features.feature['label'].float_list.value[0]
    raw_features = example.features.feature['features'].bytes_list.value[0]
    raw = bytes(raw_features)
    idx = raw.index(HEADER_MARKER)
    float_bytes = raw[idx+2 : idx+2+112]
    floats = struct.unpack('28f', float_bytes)
    row = {"label": label}
    row.update(dict(zip(FEATURE_NAMES, floats)))
    return row

def tfrecords_to_dataframe(tfrecord_paths, max_records=None):
    records = []
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    if max_records:
        dataset = dataset.take(max_records)
    for i, raw_record in enumerate(dataset):
        records.append(parse_record(raw_record))
        if (i + 1) % 10000 == 0:
            print(f"  Parsed {i+1} records...")
    return pd.DataFrame(records)

if __name__ == "__main__":
    train_files = sorted(TRAINING_DIR.glob("*.tfrecord"))
    print(f"Found {len(train_files)} training files")
    print("Parsing first 50,000 records...")
    df = tfrecords_to_dataframe(train_files, max_records=50000)
    print(f"\nShape: {df.shape}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")