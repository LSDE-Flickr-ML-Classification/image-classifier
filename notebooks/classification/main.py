import time

from image_classifier import BatchClassifier
from output_writer import OutputWriter
from input_reader import InputReader


# COMMAND ----------


INPUT_PARQUETS = ["/home/corneliu/downloaded_images-sample.parquet"]

OUTPUT_CLASSIFIED_PARQUET = "/home/corneliu/classified_images.parquet"
IMAGENET_CLASSES_LOCATION = "/media/corneliu/UbuntuFiles/projects/flickr-image-classification/scripts/classifier/imagenet_classes.json"
CHECKPOINT_LOCATION = "/tmp/classification-checkpoint"
CAN_USE_CUDA = True
BATCH_SIZE = 200
MAX_LABELS = 5
NUM_WORKERS = 1

total_images_count = 0

# COMMAND ----------


batch_classifier = BatchClassifier(
    can_use_cuda=CAN_USE_CUDA,
    classes_file_location=IMAGENET_CLASSES_LOCATION,
    batch_size=BATCH_SIZE,
    max_labels=MAX_LABELS,
    num_workers=NUM_WORKERS
)


# COMMAND ----------


output_writer = OutputWriter(OUTPUT_CLASSIFIED_PARQUET)


# COMMAND ----------


def process_dataframe(dataframe):
    global total_images_count
    start_time_batch_processing = time.time()
    batch_size = dataframe.size
    print("Classifying a Batch of size: %d" % batch_size)
    matched_labels = batch_classifier.classify_images(dataframe)
    size_matched_labels = len(matched_labels)
    print("Classified a batch of %d valid elements from %d possible elements" % (size_matched_labels, batch_size))
    output_writer.write_to_parquet(matched_labels)
    duration_batch_processing = time.time() - start_time_batch_processing
    speed = size_matched_labels / duration_batch_processing
    print("Duration of processing a batch of %d images: %.2f s. Speed: %2.f img/s" %
          (size_matched_labels, duration_batch_processing, speed))
    total_images_count = total_images_count + size_matched_labels
    print("Total images count: %d" % total_images_count)


# COMMAND ----------


def read_downloaded_images():
    input_reader = InputReader(INPUT_PARQUETS)
    input_reader.read_parquet_files_as_row_groups(callback=process_dataframe)


# COMMAND ----------


if __name__ == '__main__':
    read_downloaded_images()
