from data_handler import DataHandler
from image_downloader import ImageDownloader
from image_classifier import BatchClassifier
from itertools import islice

CLASSES_LOCATION = '../scripts/classifier/imagenet_classes.json'
# PARQUET_FILE_INPUT_LOCATION = "/home/corneliu/flickr_sampled.parquet"
PARQUET_FILE_INPUT_LOCATION = "/home/corneliu/flickr.parquet"

PARQUET_FILE_OUTPUT_LOCATION = "/home/corneliu/classification_result.parquet"

CUDA = True
SHOULD_USE_REDUCED_SAMPLED = False
SAMPLE_SIZE = 50

MAX_LABELS = 5
NUM_WORKERS = 1
BATCH_SIZE = 250

image_links = DataHandler.get_parquet_data(PARQUET_FILE_INPUT_LOCATION)

if SHOULD_USE_REDUCED_SAMPLED:
    print("Using a sample of %d items from the original parquet file" % SAMPLE_SIZE)
    image_links = list(islice(image_links, SAMPLE_SIZE))

tensor_images = ImageDownloader.download_images(image_links)

classifier = BatchClassifier(
    can_use_cuda=CUDA,
    classes_file_location=CLASSES_LOCATION,
    batch_size=BATCH_SIZE,
    max_labels=MAX_LABELS,
    num_workers=NUM_WORKERS
)

results = classifier.classify_image_tensors(tensor_images)
results_df = DataHandler.convert_classification_result_to_dataframe(results)

DataHandler.write_classification_result(results_df, PARQUET_FILE_OUTPUT_LOCATION)
