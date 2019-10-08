from pyspark.sql import SparkSession
from pyspark.sql import Row

spark = SparkSession.builder.master("local").appName("Word Count").getOrCreate()


class DataHandler:

    @staticmethod
    def get_parquet_data(parquet_file_location):
        print("Reading the input parquet file at %s" % parquet_file_location)
        data_frame = spark.read.parquet(parquet_file_location)
        links_dataframe = data_frame.select("id", "photo_video_download_url")
        return links_dataframe.collect()

    @staticmethod
    def convert_classification_result_to_dataframe(classification_result):
        print("Converting the classification result to dataframe")

        rows_list = []

        for image_id, download_link, matched_classes in classification_result:
            for label, confidence in matched_classes.items():
                rows_list.append(Row(image_id=image_id, label=label, confidence=confidence))

        return spark.createDataFrame(rows_list)

    @staticmethod
    def write_classification_result(results_df, parquet_destination):
        print("Writing the classification result to parquet file: %s" % parquet_destination)
        results_df.write.mode('overwrite').parquet(parquet_destination)


if __name__ == '__main__':
    parquet_input_file_location = "/home/corneliu/flickr_sampled.parquet"
    links_list = DataHandler.get_parquet_data(parquet_input_file_location)
    print(links_list)