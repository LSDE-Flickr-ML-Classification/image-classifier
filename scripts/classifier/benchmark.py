#!/usr/bin/env python

from torchvision import models
from torchvision import transforms
from PIL import Image
from itertools import islice
from PIL import Image

import torch
import argparse
import os
import time
import json
import subprocess

CLASSES_LOCATION = './imagenet_classes.json'

RESIZE_RESOLUTION = 256
CENTER_CROP = 224
MEAN = [0.485, 0.456, 0.406]
STANDARD_DEVIATION = [0.229, 0.224, 0.225]
CLASSIFICATION_CLASSES = 5


def is_valid_folder(parser, folder):
    if not os.path.exists(folder) and not os.path.isdir(folder):
        parser.error("The path you specified is not a folder or does not exist!")
    else:
        return folder


def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize photos from a dataset and their labels")

    parser.add_argument('-d', '--dataset',
                        dest="dataset",
                        required=True,
                        help="The folder where the photos from the dataset are located", metavar="FOLDER",
                        type=lambda x: is_valid_folder(parser, x))

    parser.add_argument('-t', '--type',
                        dest="type",
                        help="The type of device to use", metavar="DEVICE",
                        default="CPU",
                        choices=["CPU", "GPU"])

    return parser.parse_args()


def get_classes(classes_json):
    """Returns a list of classes that should be used to classify the image"""
    print("Obtaining classes stored at: %s!" % classes_json)

    if not os.path.exists(os.path.dirname(classes_json)):
        print("Could not find the classes file: %s" % classes_json)
        exit(-1)

    with open(classes_json) as json_file:
        classes_list = json.load(json_file)

    return [classes_list[str(k)][1] for k in range(len(classes_list))]


def preprocess_image(image_filename):
    """
    Resizes, crops, normalizes and transforms the image to a tensor.
    :param image_filename: Original picture without any other preprocessing
    :return: A tensor representing the input layer of the Neural Network
    """

    img = Image.open(image_filename)
    transform = transforms.Compose([
        transforms.Resize(RESIZE_RESOLUTION),
        transforms.CenterCrop(CENTER_CROP),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STANDARD_DEVIATION
        )])

    try:
        img_transformed = transform(img)
        return torch.unsqueeze(img_transformed, 0)
    except Exception as e:
        return None


def execute_classification(model, classes, tensor, output_classes):
    out = model(tensor)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return {classes[idx]: percentage[idx].item() for idx in islice(indices[0], output_classes)}


def benchmark():
    args = parse_arguments()

    device = torch.device("cuda:0" if args.type == "GPU" else "cpu")

    model = models.inception_v3(pretrained=True)
    model = model.to(device)
    model.eval()

    img_classes = get_classes(CLASSES_LOCATION)

    dash = '-' * 115

    print(dash)
    print("{:<25s} | {:>4s} | {:>8.4s} | {:>15s} | {:>15s} | {:>15s} | {:>15s}".format("Folder", "Cnt.", "Dur.",
                                                                                       "Img/sec", "Dur/Photo",
                                                                                       "Avg img/sec", "Avg dur/photo"))
    print(dash)

    total_image_count = 0
    time_start_all = time.time()

    for root, dirs, files in os.walk(args.dataset):
        path = root.split(os.sep)
        nr_photos = 0
        time_started = time.time()

        for file in files:
            full_file_path = os.path.join(root, file)
            if full_file_path.endswith(".jpg") or full_file_path.endswith(".jpeg"):
                if nr_photos >= 50:
                    break
                img_tensor = preprocess_image(full_file_path)
                if img_tensor is None:
                    continue
                img_tensor = img_tensor.to(device)
                execute_classification(model, img_classes, img_tensor, CLASSIFICATION_CLASSES)
                nr_photos = nr_photos + 1

        time_ended = time.time()
        total_image_count = total_image_count + nr_photos

        time_duration = time_ended - time_started
        time_duration_all = time_ended - time_start_all

        photos_per_second = 0 if time_duration == 0 else nr_photos / time_duration
        photos_per_second_all = 0 if time_duration_all == 0 else total_image_count / time_duration_all

        duration_per_photo = 0 if nr_photos == 0 else time_duration / nr_photos
        duration_per_photo_all = 0 if total_image_count == 0 else time_duration_all / total_image_count

        print("{:<25s} | {:>4d} | {:>8.4f} | {:>15.4f} | {:>15.4f} | {:>15.4f} | {:>15.4f}".format(path[-1] + "/",
                                                                                                   nr_photos,
                                                                                                   time_duration,
                                                                                                   photos_per_second,
                                                                                                   duration_per_photo,
                                                                                                   photos_per_second_all,
                                                                                                   duration_per_photo_all))


if __name__ == '__main__':
    benchmark()
