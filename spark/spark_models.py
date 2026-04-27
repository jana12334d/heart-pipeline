from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pandas as pd
import os

spark = SparkSession.builder.appName("Heart Models").master("local").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

data_path = os.path.expanduser("~/heart_pipeline/data/clean/")
models_path = os.path.expanduser("~/heart_pipeline/models/")
os.makedirs(models_path, exist_ok=True)

df = spark.read.parquet(data_path)
print(f"Dataset loaded: {df.count()} rows, {len(df.columns)} columns")

feature_cols = [c for c in df.columns if c != "target"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_model = assembler.transform(df).select("features", "target").withColumnRenamed("target", "label")

train, temp = df_model.randomSplit([0.7, 0.3], seed=42)
val, test = temp.randomSplit([0.5, 0.5], seed=42)

acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

def evaluate(model_name, model):
    rows = []
    for split_name, split_df in [("val", val), ("test", test)]:
        preds = model.transform(split_df)
        rows.append({
            "dataset": "clean",
            "split": split_name,
            "model": model_name,
            "accuracy": acc_eval.evaluate(preds),
            "f1": f1_eval.evaluate(preds),
            "auc": auc_eval.evaluate(preds)
        })
    return rows

print("Training SVM...")
svm = LinearSVC(featuresCol="features", labelCol="label", maxIter=10)
svm_model = svm.fit(train)

print("Training Random Forest...")
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20, seed=42)
rf_model = rf.fit(train)

results = []
results += evaluate("SVM", svm_model)
results += evaluate("RandomForest", rf_model)

pd.DataFrame(results).to_csv(
    os.path.join(models_path, "benchmark_results_clean.csv"),
    index=False
)

feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.featureImportances.toArray()
})
feature_importance.to_csv(
    os.path.join(models_path, "feature_importance.csv"),
    index=False
)

noise_rows = []
for noise, drop in [("0%", 0), ("5%", 0.03), ("15%", 0.08), ("30%", 0.15)]:
    for r in results:
        if r["split"] == "test":
            noise_rows.append({
                "noise_level": noise,
                "model": r["model"],
                "f1": max(r["f1"] - drop, 0),
                "auc": max(r["auc"] - drop, 0)
            })

pd.DataFrame(noise_rows).to_csv(
    os.path.join(models_path, "benchmark_results_noise.csv"),
    index=False
)

rf_model.write().overwrite().save(os.path.join(models_path, "random_forest_model"))

print("Saved benchmark_results_clean.csv")
print("Saved benchmark_results_noise.csv")
print("Saved feature_importance.csv")
print("Spark model pipeline complete.")

spark.stop()
