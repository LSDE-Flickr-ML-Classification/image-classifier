import pandas as pd
import torch
import requests
import io
import time

from PIL import Image
from torchvision import models, transforms

from pyspark.sql import SparkSession
from torch.utils.data import Dataset


# COMMAND ----------


CUDA = False
COMPUTE_DEVICE = "cpu"
BC_MODEL_STATE = None


# COMMAND ----------


spark = SparkSession.builder.appName("flickr-image-classifier").getOrCreate()
sc = spark.sparkContext


# COMMAND ----------


class ImageDataset(Dataset):
    def __init__(self, images_as_bytes, transform=None):
        self.images_as_bytes = images_as_bytes
        self.transform = transform

    def __len__(self):
        return len(self.images_as_bytes)

    def __getitem__(self, item):
        image = Image.open(self.images_as_bytes[item])
        if self.transform is not None:
            image = self.transform(image)
        return image


# COMMAND ----------


def get_model_for_evaluation(bc_model_state):
    model = models.inception_v3(pretrained=True)
    model.load_state_dict(bc_model_state.value)
    model.eval()
    return model


# COMMAND ----------


def predict_batch(image_bytes):
    global BC_MODEL_STATE
    global COMPUTE_DEVICE

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    nr_images = len(image_bytes)
    print("Starting prediction of %d images" % nr_images)

    images = ImageDataset(image_bytes, transform=transform)
    loader = torch.utils.data.DataLoader(images, batch_size=250, num_workers=1)

    model = get_model_for_evaluation(BC_MODEL_STATE)
    model.to(COMPUTE_DEVICE)

    start_time_classification = time.time()

    all_predictions = []
    with torch.no_grad():
        for batch in loader:
            predictions = list(model(batch.to(COMPUTE_DEVICE)).cpu().numpy())
            for prediction in predictions:
                all_predictions.append(prediction)

    classification_duration = time.time() - start_time_classification
    images_per_second = nr_images / classification_duration
    time_per_image = classification_duration / nr_images

    print("Duration: %.4f | Nr images: %d | Images/s: %.4f | Classification time per image: %.4f" % (
        classification_duration, nr_images, images_per_second, time_per_image
    ))

    return pd.Series(all_predictions)


# COMMAND ----------


def setup_compute_device():
    global COMPUTE_DEVICE
    use_cuda = CUDA and torch.cuda.is_available()
    COMPUTE_DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print("Using the following device for classification: %s" % COMPUTE_DEVICE)


# COMMAND ----------


def broadcast_model_to_nodes():
    global BC_MODEL_STATE
    BC_MODEL_STATE = sc.broadcast(models.inception_v3(pretrained=True).state_dict())
    print("Broadcasted the model to all worker nodes")


# COMMAND ----------


def download_images_as_bytes(image_urls):
    print("Downloading %d possible images..." % len(image_urls))
    image_bytes = []
    for i in range(0, len(image_urls)):
        url = image_urls[i]
        print("Downloading image: %d, url: %s" % (i, url))

        response = requests.get(url)
        if response.status_code != 200:
            print("Could not get image for url: %s" % url)
            continue

        image_bytes.append(io.BytesIO(response.content))

    print("Succefuly Downloaded %d images" % len(image_bytes))
    return image_bytes


# COMMAND ----------


def obtain_image_urls_list():
    return list_image_urls


# COMMAND ----------


setup_compute_device()
broadcast_model_to_nodes()

image_urls = obtain_image_urls_list()
image_bytes = download_images_as_bytes(image_urls)

prediction_results = predict_batch(pd.Series(image_bytes))


# COMMAND ----------










