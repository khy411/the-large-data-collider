import tensorflow as tf
import numpy as np
import struct

sample_file = ".../higgs-boson/training/shard_00.tfrecord"
raw_dataset = tf.data.TFRecordDataset(sample_file)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    raw_features = example.features.feature['features'].bytes_list.value[0]

    print(f"Bytes: {list(raw_features)}")
    print()

    raw = bytes(raw_features)
    idx = raw.index(b'\x22\x70')  # 34, 112
    float_bytes = raw[idx+2 : idx+2+112]
    
    print(f"Float bytes length: {len(float_bytes)}")
    floats = struct.unpack(f'{len(float_bytes)//4}f', float_bytes)
    print(f"Number of floats: {len(floats)}")
    print(f"Values:")
    for i, v in enumerate(floats):
        print(f"  feature_{i:02d}: {v:.6f}")
