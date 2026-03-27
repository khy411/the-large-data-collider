import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, OUTPUT_DIR

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def build_spark():
    return SparkSession.builder \
        .appName("LargeDataCollider-Classifier") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.ui.enabled", "false") \
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC") \
        .getOrCreate()

def load_data(spark, sample=True):
    # load from parquet, use sample for speed during dev
    if sample:
        path = str(PROCESSED_DIR / "higgs_train_sample.parquet")
        print("loading sample parquet (50k records)...")
    else:
        path = str(PROCESSED_DIR / "higgs_train_spark")
        print("loading full parquet (10.5M records)...")

    df = spark.read.parquet(path)

    # assemble features into a single vector column for spark ml
    assembler = VectorAssembler(inputCols=FEATURE_NAMES, outputCol="features")
    df = assembler.transform(df).select("features", "label")
    return df

def train_and_evaluate(spark, sample=True):
    df = load_data(spark, sample=sample)

    # 80/20 train/test split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"train size: {train_df.count():,} | test size: {test_df.count():,}")

    # random forest classifier
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=10,
        seed=42
    )

    print("\ntraining random forest...")
    model = rf.fit(train_df)

    print("evaluating...")
    predictions = model.transform(test_df)

    # auc roc
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc = auc_evaluator.evaluate(predictions)

    # accuracy
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = acc_evaluator.evaluate(predictions)

    # f1
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    f1 = f1_evaluator.evaluate(predictions)

    print(f"\n  auc roc  : {auc:.4f}")
    print(f"  accuracy : {accuracy:.4f}")
    print(f"  f1 score : {f1:.4f}")

    return model, predictions, auc, accuracy, f1

def plot_feature_importance(model, auc):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # extract feature importances from random forest
    importances = model.featureImportances.toArray()
    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": importances
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))
    bars = ax.barh(
        importance_df["feature"],
        importance_df["importance"],
        color=["darkviolet" if v >= importance_df["importance"].median()
               else "dodgerblue" for v in importance_df["importance"]],
        alpha=0.85,
        edgecolor="white"
    )
    ax.set_xlabel("Feature Importance (Gini Impurity)", fontsize=11)
    ax.set_title(f"Random Forest Feature Importances\nAUC-ROC: {auc:.4f}",
                 fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=9)

    for bar, val in zip(bars, importance_df["importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved: {out_path}")
    plt.close()

def plot_importance_vs_auc(model):
    # compare model feature importance vs auc separation scores from analytics
    from sklearn.metrics import roc_auc_score
    import pandas as pd_local

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    importances = model.featureImportances.toArray()

    # load pandas parquet for auc computation
    pdf = pd_local.read_parquet(str(PROCESSED_DIR / "higgs_train_sample.parquet"))
    auc_scores = {}
    for feat in FEATURE_NAMES:
        auc = roc_auc_score(pdf["label"], pdf[feat])
        auc_scores[feat] = max(auc, 1 - auc)

    compare_df = pd_local.DataFrame({
        "feature": FEATURE_NAMES,
        "model_importance": importances,
        "auc_score": [auc_scores[f] for f in FEATURE_NAMES]
    }).sort_values("model_importance", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(FEATURE_NAMES))
    width = 0.35

    ax.bar(x - width/2, compare_df["model_importance"],
           width, label="RF Importance", color="darkviolet", alpha=0.8)
    ax.bar(x + width/2, compare_df["auc_score"] - 0.5,
           width, label="AUC Score (above 0.5)", color="dodgerblue", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(compare_df["feature"], rotation=45, ha="right", fontsize=8)
    ax.set_title("Model Feature Importance vs AUC Separation Score",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "importance_vs_auc.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")

    # set sample=False to train on full 10.5M records (takes longer)
    model, predictions, auc, accuracy, f1 = train_and_evaluate(spark, sample=False)

    print("\ngenerating plots...")
    plot_feature_importance(model, auc)
    plot_importance_vs_auc(model)

    spark.stop()
    print("\nclassifier complete")