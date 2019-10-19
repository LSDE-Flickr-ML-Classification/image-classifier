from pyspark.sql import Row
from pyspark.sql import SparkSession
import time


# COMMAND ----------


spark = SparkSession.builder.master("local[*]").appName("Images downloader").getOrCreate()
sc = spark.sparkContext


# COMMAND ----------


class OutputWriter:
    def __init__(self, output_parquet):
        self.output_parquet = output_parquet

    def write_to_parquet(self, classification_result):
        start_time_writing = time.time()
        classification_df = OutputWriter.convert_classification_to_dataframe(classification_result)
        classification_df.write.mode('append').parquet(self.output_parquet)
        duration_writing = time.time() - start_time_writing

        print("Wrote the classification result to the parquet file: %s. Duration: %.2f" %
              (self.output_parquet, duration_writing))

    @staticmethod
    def convert_classification_to_dataframe(classification_result):
        print("Converting the classification result to dataframe")

        rows_list = []

        for image_id, matched_classes in classification_result:
            for label, confidence in matched_classes.items():
                rows_list.append(Row(image_id=image_id, label=label, confidence=confidence))

        return spark.createDataFrame(rows_list)
