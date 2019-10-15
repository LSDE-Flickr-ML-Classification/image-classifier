# %run /group07/downloader/image_downloader
from image_downloader import ImageDownloader


# COMMAND ----------


from pyspark.sql import SparkSession
import torch
import time
import io


# COMMAND ----------


INPUT_PARQUET_LOCATION = "/home/corneliu/flickr_sampled.parquet/"
TENSORS_OUTPUT_LOCATION = "/home/corneliu/tensors"


# COMMAND ----------


spark = SparkSession.builder.master("local[*]").appName("Images downloader").getOrCreate()
sc = spark.sparkContext
count_accumulator = sc.accumulator(0)


# COMMAND ----------


def process_data_row(row, counter):
    image_download_link = row.photo_video_download_url
    image_id = row.id
    image_as_tensor = ImageDownloader.get_normalized_photo(image_download_link)

    if image_as_tensor is None:
        return

    counter.add(1)

    tensor_save_location = TENSORS_OUTPUT_LOCATION + "/" + str(image_id)
    torch.save(image_as_tensor, tensor_save_location)




# COMMAND ----------

time_started = time.time()
print("Default parallelism: %d" % sc.defaultParallelism)
flickr_image_df = spark.read.parquet(INPUT_PARQUET_LOCATION).select("id", "photo_video_download_url")
print("Number of partitions: %d" % flickr_image_df.rdd.getNumPartitions())
flickr_image_df.foreach(lambda row: process_data_row(row, count_accumulator))
time_taken = time.time() - time_started
print("Total number of images: %d. Time taken: %.2f s" % (count_accumulator.value, time_taken))



