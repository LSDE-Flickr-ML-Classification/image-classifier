from pyspark.sql.types import BinaryType, StructType, StringType, StructField
import time


# COMMAND ----------


from image_classifier import BatchClassifier
from output_writer import OutputWriter
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName("Images downloader").getOrCreate()
sc = spark.sparkContext


# COMMAND ----------


INPUT_IMAGES_PARQUET = "/home/corneliu/downloaded_images/downloaded_images-*.parquet"
OUTPUT_CLASSIFIED_PARQUET = "/home/corneliu/classified_images.parquet"
IMAGENET_CLASSES_LOCATION = "/media/corneliu/UbuntuFiles/projects/flickr-image-classification/scripts/classifier/imagenet_classes.json"
CHECKPOINT_LOCATION = "/tmp/classification-checkpoint"
CAN_USE_CUDA = True
BATCH_SIZE = 200
MAX_LABELS = 5
NUM_WORKERS = 1


# COMMAND ----------


def process_batch(df, batch_classifier, output_writer):
    start_time_batch_processing = time.time()
    batch_size = df.count()
    print("Classifying a Batch of size: %d" % batch_size)
    images_batch = df.collect()
    matched_labels = batch_classifier.classify_images(images_batch)
    size_matched_labels = len(matched_labels)
    print("Classified a batch of %d valid elements from %d possible elements" % (size_matched_labels, batch_size))
    output_writer.write_to_parquet(matched_labels)
    duration_batch_processing = time.time() - start_time_batch_processing
    speed = size_matched_labels / duration_batch_processing
    print("Duration of processing a batch of %d images: %.2f s. Speed: %2.f img/s" %
          (size_matched_labels, duration_batch_processing, speed))


# COMMAND ----------


def read_downloaded_images():
    input_parquet_schema = StructType([
        StructField("id", StringType()),
        StructField("image_bytes", BinaryType())
    ])

    batch_classifier = BatchClassifier(
        can_use_cuda=True,
        classes_file_location=IMAGENET_CLASSES_LOCATION,
        batch_size=200,
        max_labels=5,
        num_workers=1
    )

    output_writer = OutputWriter(OUTPUT_CLASSIFIED_PARQUET)

    parquet_read_stream = spark\
        .readStream \
        .schema(input_parquet_schema)\
        .parquet(INPUT_IMAGES_PARQUET)\

    parquet_read_stream.writeStream \
        .option("checkpointLocation", CHECKPOINT_LOCATION)\
        .foreachBatch(lambda df, epoch_id: process_batch(df, batch_classifier, output_writer)) \
        .start()\
        .processAllAvailable()


# COMMAND ----------


read_downloaded_images()
