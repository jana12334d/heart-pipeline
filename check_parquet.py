from pyspark.sql import SparkSession
import glob, os

spark = SparkSession.builder \
    .appName("check") \
    .master("local") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Try main path (if exists)
base = os.path.expanduser('~/heart_pipeline/hdfs_output/clean/')
dirs = glob.glob(base + '*')

if dirs:
    df = spark.read.parquet(dirs[0])
    print("Clean rows:", df.count())
    print("Columns:", df.columns)
    df.show(3, truncate=True)

else:
    # fallback to your actual saved location
    base2 = os.path.expanduser('~/heart_pipeline/')
    dirs2 = glob.glob(base2 + '**/*.parquet', recursive=True)

    if dirs2:
        df = spark.read.parquet(os.path.dirname(dirs2[0]))
        print("Rows found:", df.count())
        print("Columns:", df.columns)
        df.show(3, truncate=True)
    else:
        print("No parquet files found - check ETL output")

spark.stop()
