import time

from data_handler import DataHandler
from image_downloader import ImageDownloader
from image_classifier import BatchClassifier
from itertools import islice

CLASSES_LOCATION = '../scripts/classifier/imagenet_classes.json'
PARQUET_FILE_INPUT_LOCATION = "/home/corneliu/flickr.parquet"

PARQUET_FILE_OUTPUT_LOCATION = "/home/corneliu/classification_result.parquet"

CUDA = True
SHOULD_USE_REDUCED_SAMPLED = False
SAMPLE_SIZE = 50

MAX_LABELS = 5
NUM_WORKERS = 1
NUM_DOWNLOAD_THREADS = 8
BATCH_SIZE = 250


def download_and_classify_in_batches(complete_links_list, classifier):
    print("Total amount of images to be downloaded and classified: %d" % len(complete_links_list))

    for index in range(0, len(complete_links_list), BATCH_SIZE):
        time_start = time.time()
        print("Downloading and classifying batch: %d -> %d" % (index, index + BATCH_SIZE))

        links_batch = complete_links_list[index:index+BATCH_SIZE]
        tensor_images = ImageDownloader.download_images(links_batch, NUM_DOWNLOAD_THREADS)

        if len(tensor_images) == 0:
            print("Skipping classification of empy list")
            continue

        results = classifier.classify_image_tensors(tensor_images)
        results_df = DataHandler.convert_classification_result_to_dataframe(results)
        DataHandler.write_classification_result(results_df, PARQUET_FILE_OUTPUT_LOCATION)

        duration = time.time() - time_start
        print("Duration of donwloading and classification for batch: %.2f" % duration)


def execute():
    classifier = BatchClassifier(
        can_use_cuda=CUDA,
        classes_file_location=CLASSES_LOCATION,
        batch_size=BATCH_SIZE,
        max_labels=MAX_LABELS,
        num_workers=NUM_WORKERS
    )

    image_links = DataHandler.get_unprocessed_links(PARQUET_FILE_INPUT_LOCATION, PARQUET_FILE_OUTPUT_LOCATION)

    if SHOULD_USE_REDUCED_SAMPLED:
        print("Using a sample of %d items from the original parquet file" % SAMPLE_SIZE)
        image_links = list(islice(image_links, SAMPLE_SIZE))

    download_and_classify_in_batches(image_links, classifier)


execute()












