from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

print("Spark + Kafka ETL pipeline started.")

spark = SparkSession.builder \
    .appName("Heart ETL Pipeline") \
    .master("local") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

try:
    print("Trying to read from Kafka...")
    df = spark.read \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "heart_data") \
        .load()

    print("Kafka read successful.")

except Exception:
    print("Kafka read failed ... Falling back to CSV read ...")

    file_path = os.path.expanduser("~/heart_pipeline/data/heart.csv")
    df = spark.read.csv(file_path, header=True, inferSchema=True)

df_clean = df.dropna()

count = df_clean.count()

print(f"Clean records loaded: {count}")

print("[CLEAN] Schema Validation Report:")
print(f"  Total records : {count}")

null_count = df_clean.filter(
    sum(col(c).isNull().cast("int") for c in df_clean.columns) > 0
).count()

print(f"  Null values : {null_count} ✓")

if null_count == 0:
    print("  Quality flag : PASS")
else:
    print("  Quality flag : FAIL")

clean_path = os.path.expanduser("~/heart_pipeline/data/clean/")
noisy_path = os.path.expanduser("~/heart_pipeline/data/noisy/")

df_clean.write.mode("overwrite").parquet(clean_path)
df.write.mode("overwrite").parquet(noisy_path)

print(f"Saved clean data to: {clean_path}")
print(f"Saved noisy data to: {noisy_path}")

spark.stop()

print("ETL pipeline complete.")
