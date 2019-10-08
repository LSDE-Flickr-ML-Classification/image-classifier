from torch.utils.data import Dataset
from torchvision import models
from itertools import islice

import time
import torch
import os
import json


class ImageDataset(Dataset):
    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, index):
        item = self.batch[index]
        return item[0], item[1], item[2]


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

    def get_matched_labels(self, predictions, image_ids, image_download_links):
        results = []

        for index in range(0, len(predictions)):
            prediction = predictions[index]

            percentage = torch.nn.functional.softmax(prediction, dim=0) * 100
            _, indices = torch.sort(prediction, descending=True)

            matched_classes = {self.classes[idx]: percentage[idx].item() for idx in islice(indices, self.max_labels)}
            image_id = image_ids[index]
            image_download_url = image_download_links[index]

            results.append((image_id, image_download_url, matched_classes))

        return results

    def classify_image_tensors(self, batch):
        image_tensors = ImageDataset(batch)
        loader = torch.utils.data.DataLoader(image_tensors, batch_size=self.batch_size, num_workers=self.num_workers)
        results = []

        print("Starting classifying the images")

        with torch.no_grad():
            for batch in loader:

                start_time_batch = time.time()

                image_ids = batch[0]
                image_download_links = batch[1]
                image_tensors = batch[2]

                predictions = self.model(image_tensors.to(self.compute_device)).cpu()
                batch_results = self.get_matched_labels(predictions, image_ids, image_download_links)
                results.extend(batch_results)

                duration_batch = time.time() - start_time_batch
                print("Classified a batch of %d elements in %.2f seconds" % (len(image_ids), duration_batch))

        print("Finished classifying all the batches")

        return results








