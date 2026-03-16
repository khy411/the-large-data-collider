import struct
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TRAINING_DIR, VALIDATION_DIR, PROCESSED_DIR

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField, FloatType, IntegerType
)
import tensorflow as tf

# all 28 higgs boson feature names in order
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

# protobuf byte marker that precedes the 112 float bytes
HEADER_MARKER = b'\x22\x70'


def build_schema():
    # define spark schema: label as int, all features as float
    fields = [StructField("label", IntegerType(), False)]
    for name in FEATURE_NAMES:
        fields.append(StructField(name, FloatType(), False))
    return StructType(fields)


def parse_tfrecord_file(path):
    # runs on each spark worker, parses one tfrecord file into rows
    rows = []
    try:
        dataset = tf.data.TFRecordDataset(str(path))
        for raw_record in dataset:
            try:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())

                # skip anything thats not a valid label
                label = int(example.features.feature['label'].float_list.value[0])
                if label not in (0, 1):
                    continue

                # unpack the nested protobuf bytes into 28 float32 values
                raw_features = example.features.feature['features'].bytes_list.value[0]
                raw = bytes(raw_features)
                idx = raw.index(HEADER_MARKER)
                float_bytes = raw[idx+2 : idx+2+112]
                floats = list(struct.unpack('28f', float_bytes))

                # negative pt values are sentinels meaning no jet detected, zero them out
                for i, name in enumerate(FEATURE_NAMES):
                    if name.endswith('_pt') and floats[i] < 0:
                        floats[i] = 0.0

                rows.append(tuple([label] + floats))
            except Exception:
                continue
    except Exception:
        pass
    return rows


def run_spark_pipeline(mode="sample"):
    # sample mode runs on 1 file only, switch to full for all 11m records
    spark = SparkSession.builder \
        .appName("LargeDataCollider") \
        .master("local[2]") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.ui.enabled", "false") \
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print(f"spark version: {spark.version}")

    train_files = sorted(TRAINING_DIR.glob("*.tfrecord"))
    val_files = sorted(VALIDATION_DIR.glob("*.tfrecord"))

    if mode == "sample":
        train_files = train_files[:1]
        val_files = val_files[:1]
        print(f"sample mode: {len(train_files)} train file, {len(val_files)} val file")
    else:
        print(f"full mode: {len(train_files)} train files, {len(val_files)} val files")

    schema = build_schema()

    for split, files in [("train", train_files), ("validation", val_files)]:
        print(f"\nprocessing {split}...")

        # distribute file list across workers
        files_rdd = spark.sparkContext.parallelize(
            [str(f) for f in files], numSlices=len(files)
        )

        # parse each file in parallel
        rows_rdd = files_rdd.flatMap(parse_tfrecord_file)
        df = spark.createDataFrame(rows_rdd, schema=schema)

        total = df.count()
        signal = df.filter(col("label") == 1).count()
        background = df.filter(col("label") == 0).count()

        print(f"  total rows : {total:,}")
        print(f"  signal     : {signal:,} ({signal/total*100:.1f}%)")
        print(f"  background : {background:,} ({background/total*100:.1f}%)")

        # save as partitioned parquet
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = str(PROCESSED_DIR / f"higgs_{split}_spark")
        df.write.mode("overwrite").parquet(out_path)
        print(f"  saved: {out_path}/")

        df.select("label", "lepton_pT", "m_bb", "m_wwbb").describe().show()

    spark.stop()
    print("\nspark pipeline complete")


if __name__ == "__main__":
    run_spark_pipeline(mode="full")