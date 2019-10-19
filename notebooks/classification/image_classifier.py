from torch.utils.data import Dataset
from torchvision import models
from itertools import islice
from PIL import Image
from torchvision import transforms

import time
import torch
import os
import json
import io


# COMMAND ----------


MEAN = [0.485, 0.456, 0.406]
STANDARD_DEVIATION = [0.229, 0.224, 0.225]
IMAGE_TRANSFORM_FUNCTION = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STANDARD_DEVIATION)
])


# COMMAND ----------


class ImageDataset(Dataset):
    def __init__(self, batch):
        self.batch = ImageDataset.preprocess_dataset(batch)

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, index):
        return self.batch[index][0], self.batch[index][1]

    @staticmethod
    def preprocess_dataset(batch_dataset):
        image_tensors = []
        for row in batch_dataset:
            image_id = row.id
            if row.image_bytes is None:
                continue
            image_bytes = bytearray(row.image_bytes)
            image = Image.open(io.BytesIO(image_bytes))
            try:
                image_tensor = IMAGE_TRANSFORM_FUNCTION(image)
                image_tensors.append((image_id, image_tensor))
            except Exception as e:
                # TODO: Verify for greyscale images
                continue

        return image_tensors


# COMMAND ----------


class BatchClassifier:
    def __init__(self, can_use_cuda, classes_file_location, batch_size, max_labels, num_workers):
        self.batch_size = batch_size
        self.max_labels = max_labels
        self.num_workers = num_workers

        self.compute_device = BatchClassifier.get_compute_device(can_use_cuda)
        self.model = BatchClassifier.prepare_model_for_inference(self.compute_device)
        self.classes = BatchClassifier.get_classes(classes_file_location)

    @staticmethod
    def prepare_model_for_inference(compute_device):
        print("Initializing the classification model")
        model = models.inception_v3(pretrained=True)
        model.eval()
        model.to(compute_device)
        return model

    @staticmethod
    def get_compute_device(can_use_cuda):
        is_cuda_usable = can_use_cuda and torch.cuda.is_available()
        compute_device = torch.device("cuda" if is_cuda_usable else "cpu")
        print("Using the following device for classification: %s" % compute_device)
        return compute_device

    @staticmethod
    def get_classes(classes_file_location):
        print("Retrieving the classification classes at: %s" % classes_file_location)

        if not os.path.exists(os.path.dirname(classes_file_location)):
            print("Could not find the classes file: %s" % classes_file_location)
            exit(-1)

        with open(classes_file_location) as json_file:
            classes_list = json.load(json_file)

        return [classes_list[str(k)][1] for k in range(len(classes_list))]

    def get_matched_labels(self, predictions, image_ids):
        results = []

        for index in range(0, len(predictions)):
            prediction = predictions[index]

            percentage = torch.nn.functional.softmax(prediction, dim=0) * 100
            _, indices = torch.sort(prediction, descending=True)

            matched_classes = {self.classes[idx]: percentage[idx].item() for idx in islice(indices, self.max_labels)}
            image_id = image_ids[index]

            results.append((image_id, matched_classes))

        return results

    def classify_images(self, batch):
        start_time_classification = time.time()

        tensors_dataset = ImageDataset(batch)

        loader = torch.utils.data.DataLoader(tensors_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        results = []

        print("Starting classifying %d images" % len(tensors_dataset))

        with torch.no_grad():
            for batch in loader:
                start_time_batch = time.time()

                image_ids = batch[0]
                image_tensors = batch[1]

                print("Classifying a batch of %d images" % len(image_ids))

                predictions = self.model(image_tensors.to(self.compute_device)).cpu()
                batch_results = self.get_matched_labels(predictions, image_ids)
                results.extend(batch_results)

                duration_batch = time.time() - start_time_batch
                print("Classified a batch of %d elements in %.2f seconds" % (len(image_ids), duration_batch))

        duration_classification = time.time() - start_time_classification
        print("Finished classifying the images. Duration: %.2f" % duration_classification)

        return results
