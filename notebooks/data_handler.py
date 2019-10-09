from pyspark.sql import SparkSession
from pyspark.sql import Row

import os

spark = SparkSession.builder.master("local").appName("Word Count").getOrCreate()


class DataHandler:

    @staticmethod
    def get_unprocessed_links(input_parqut, output_parquet):
        print("Reading the input parquet file at %s" % input_parqut)

        links_dataframe = spark.read.parquet(input_parqut).select("id", "photo_video_download_url")

        if not os.path.exists(output_parquet):
            print("No output parquet exists, processing the entire dataset!")
            return links_dataframe.collect()

        print("Found output parquet, removing the already processed links")
        classified_ids = spark.read.parquet(output_parquet)

        unprocessed_links = links_dataframe\
            .join(classified_ids, links_dataframe.id == classified_ids.image_id, how='left_anti')\
            .select("id", "photo_video_download_url")

        return unprocessed_links.collect()

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
        results_df.write.mode('append').parquet(parquet_destination)


if __name__ == '__main__':
    parquet_input_file_location = "/home/corneliu/flickr_sampled.parquet"
    links_list = DataHandler.get_parquet_data(parquet_input_file_location)
    print(links_list)