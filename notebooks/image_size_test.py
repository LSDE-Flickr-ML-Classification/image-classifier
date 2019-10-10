from image_downloader import ImageDownloader
from image_classifier import BatchClassifier
from data_handler import DataHandler

import io
import torch
import sys

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

links_row = DataHandler.get_unprocessed_links(PARQUET_FILE_INPUT_LOCATION, "saf")[:20]
# download_links = [row.photo_video_download_url for row in links]
# print(download_links)

for row in links_row:
    image_id, download_url, image_as_tensor = ImageDownloader.download_and_preprocess_image(row)
    if image_as_tensor is None:
        continue

    buffer = io.BytesIO()
    torch.save(image_as_tensor, buffer)