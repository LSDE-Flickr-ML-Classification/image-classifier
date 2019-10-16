# %run /group07/downloader/image_downloader
from image_downloader import ImageDownloader
from pyspark.sql.types import BinaryType


# COMMAND ----------


from pyspark.sql import SparkSession

import time
import io


# COMMAND ----------


INPUT_PARQUET_LOCATION = "/home/corneliu/flickr_sampled.parquet/"
OUTPUT_PARQUET_LOCATION = "/home/corneliu/downloaded_images.parquet"


# COMMAND ----------


spark = SparkSession.builder.master("local[*]").appName("Images downloader").getOrCreate()
sc = spark.sparkContext


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


def get_downloaded_images_df(input_parquet, sum_accumulator):
    download_image_udf = spark.udf.register("download_image", lambda x: download_image(x, sum_accumulator), BinaryType())

    downloaded_images_df = spark.read.parquet(input_parquet) \
        .select("id", "photo_video_download_url") \
        .withColumn('image_bytes', download_image_udf('photo_video_download_url')) \
        .drop('photo_video_download_url')

    return downloaded_images_df


# COMMAND ----------


def download_and_store_all_images(input_parquet, output_parquet):
    time_started = time.time()

    print("Default parallelism: %d" % sc.defaultParallelism)

    count_accumulator = sc.accumulator(0)
    downloaded_images_df = get_downloaded_images_df(input_parquet, count_accumulator)

    print("Number of partitions: %d" % downloaded_images_df.rdd.getNumPartitions())
    downloaded_images_df.write.mode('overwrite').parquet(output_parquet)

    time_taken = time.time() - time_started
    total_images_downloaded = count_accumulator.value
    images_per_second = total_images_downloaded / time_taken

    print("Total number of images: %d. Time taken: %.2f s Images/s: %.2f" %
          (total_images_downloaded, time_taken, images_per_second))


# COMMAND ----------


download_and_store_all_images(INPUT_PARQUET_LOCATION, OUTPUT_PARQUET_LOCATION)




