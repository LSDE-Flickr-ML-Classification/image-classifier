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

    location = args.dataset

    nr_photos = 0
    total_preprocessing_time = 0
    total_classification_time = 0
    total_width = 0
    total_height = 0
    img_classes = get_classes(CLASSES_LOCATION)

    for file in os.listdir(location):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            file_path = os.path.join(location, filename)
            nr_photos = nr_photos + 1

            im = Image.open(file_path)
            width, height = im.size

            total_width = total_width + width
            total_height = total_height + height

            preprocessing_start = time.time()
            img_tensor = preprocess_image(file_path)
            img_tensor = img_tensor.to(device)
            preprocessing_end = time.time()

            if img_tensor is None:
                print("Skipping image %s" % filename)
                continue

            total_preprocessing_time = total_preprocessing_time + (preprocessing_end - preprocessing_start)

            classification_start = time.time()
            matched_labels = execute_classification(model, img_classes, img_tensor, CLASSIFICATION_CLASSES)
            classification_end = time.time()

            total_classification_time = total_classification_time + (classification_end - classification_start)

            print(matched_labels, filename)

            print('{:<25s}{:5>d}'.format("Nr Photos:", nr_photos))
            print('{:<25s}{:5>.4f}'.format("Avg preprocessing time:", (total_preprocessing_time / nr_photos)))
            print('{:<25s}{:5>.4f}'.format("Avg classification time:", (total_classification_time / nr_photos)))
            print('{:<25s}{:5>.4f}'.format("Avg Width:", (total_width / nr_photos)))
            print('{:<25s}{:5>.4f}'.format("Avg Height:", (total_height / nr_photos)))


if __name__ == '__main__':
    benchmark()
