from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("Word Count").getOrCreate()

PARQUET_FILE = "/home/corneliu/classification_result.parquet"

data_frame = spark.read.parquet(PARQUET_FILE)
data_frame.show(300)
count = data_frame.count()
print(count)