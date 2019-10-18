# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import BinaryType, StructType, StringType, StructField
from image_classifier import BatchClassifier
from output_writer import OutputWriter

import time

INPUT_IMAGES_PARQUET = "/home/corneliu/downloaded_images.parquet"
CLASSIFIED_IMAGES_PARQUET = "/home/corneliu/classified_images.parquet"
IMAGENET_CLASSES_LOCATION = "/media/corneliu/UbuntuFiles/projects/flickr-image-classification/scripts/classifier/imagenet_classes.json"
CHECKPOINT_LOCATION = "/tmp/classification-checkpoint"

spark = SparkSession.builder.master("local[*]").appName("Images downloader").getOrCreate()
sc = spark.sparkContext


def process_batch(df, batch_classifier, output_writer):
    start_time_batch_processing = time.time()
    batch_size = df.count()
    print("Classifying a Batch of size: %d" % batch_size)
    images_batch = df.collect()
    matched_labels = batch_classifier.classify_images(images_batch)
    print("Classified a batch of %d valid elements from %d possible elements" % (len(matched_labels), batch_size))
    output_writer.write_to_parquet(matched_labels)
    duration_batch_processing = time.time() - start_time_batch_processing
    print("Duration of processing a batch of %d possible images: %.2f" % (batch_size, duration_batch_processing))


def read_downloaded_images(input_parquet, output_parquet, checkpoint_location, classes_location):
    input_parquet_schema = StructType([
        StructField("id", StringType()),
        StructField("image_bytes", BinaryType())
    ])

    batch_classifier = BatchClassifier(
        can_use_cuda=True,
        classes_file_location=classes_location,
        batch_size=200,
        max_labels=5,
        num_workers=1
    )

    output_writer = OutputWriter(output_parquet)

    parquet_read_stream = spark\
        .readStream \
        .schema(input_parquet_schema)\
        .parquet(input_parquet)\

    parquet_read_stream.writeStream \
        .option("checkpointLocation", checkpoint_location)\
        .foreachBatch(lambda df, epoch_id: process_batch(df, batch_classifier, output_writer)) \
        .start()\
        .processAllAvailable()


read_downloaded_images(INPUT_IMAGES_PARQUET, CLASSIFIED_IMAGES_PARQUET, CHECKPOINT_LOCATION, IMAGENET_CLASSES_LOCATION)
