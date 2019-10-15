# %run /group07/downloader/image_downloader
from image_downloader import ImageDownloader
from pyspark.sql.types import StructField, BinaryType, StructType
import pickle

# COMMAND ----------


from pyspark.sql import SparkSession
from pyspark.sql.functions import udf

import torch
import time
import io


# COMMAND ----------


INPUT_PARQUET_LOCATION = "/home/corneliu/flickr_sampled.parquet/"
IMAGES_OUTPUT_LOCATION = "/home/corneliu/downloaded_images"


# COMMAND ----------


spark = SparkSession.builder.master("local[*]").appName("Images downloader").getOrCreate()
sc = spark.sparkContext
count_accumulator = sc.accumulator(0)


# COMMAND ----------


def download_image(photo_video_download_url, count):
    downloaded_photo = ImageDownloader.get_image_from_link(photo_video_download_url)

    if downloaded_photo is None:
        return None

    count.add(1)

    buffer = io.BytesIO()
    downloaded_photo.save(buffer, format='PNG')

    return buffer.getvalue()

# COMMAND ----------


download_image_udf = spark.udf.register("download_image", lambda x: download_image(x, count_accumulator), BinaryType())

time_started = time.time()

print("Default parallelism: %d" % sc.defaultParallelism)

flickr_image_df = spark.read.parquet(INPUT_PARQUET_LOCATION)\
    .select("id", "photo_video_download_url")\
    .withColumn('image_bytes', download_image_udf('photo_video_download_url'))\
    .drop('photo_video_download_url')

print("Number of partitions: %d" % flickr_image_df.rdd.getNumPartitions())

# flickr_image_df.show(1000)

flickr_image_df.write.mode('overwrite').parquet("images.parquet")
# flickr_image_df.foreach(lambda row: process_data_row(row, count_accumulator))

time_taken = time.time() - time_started

print("Total number of images: %d. Time taken: %.2f s" % (count_accumulator.value, time_taken))



