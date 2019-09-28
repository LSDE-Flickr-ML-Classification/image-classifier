#!/usr/bin/env python

from torchvision import models
from torchvision import transforms
from PIL import Image
from itertools import islice

import torch
import json
import argparse
import os
import time
import errno


CLASSES_LOCATION = './imagenet_classes.json'
DEFAULT_NR_CLASSES = 5

RESIZE_RESOLUTION = 256
CENTER_CROP = 224
MEAN = [0.485, 0.456, 0.406]
STANDARD_DEVIATION = [0.229, 0.224, 0.225]

OUTPUT_FILE = 'out/classification-out.json'


def is_valid_file(parser, filename):
    if not os.path.exists(filename):
        parser.error("The file %s does not exist!" % filename)
    else:
        return filename  # return an open file handle


def parse_arguments():
    parser = argparse.ArgumentParser(description="Photo location")

    parser.add_argument('-f', '--filename',
                        dest="filename",
                        required=True,
                        help="The photo that will be classified", metavar="FILE",
                        type= lambda x: is_valid_file(parser, x))

    parser.add_argument('-c', '--classes',
                        dest="count_classes",
                        help="The number of classes to be returned", metavar="NR CLASSES",
                        default=DEFAULT_NR_CLASSES,
                        type=int)

    return parser.parse_args()


def preprocess_image(image_filename):
    """
    Resizes, crops, normalizes and transforms the image to a tensor.
    :param image_filename: Original picture without any other preprocessing
    :return: A tensor representing the input layer of the Neural Network
    """
    # print("Pre-processing image")

    preprocessing_start_time = time.time()

    img = Image.open(image_filename)

    transform = transforms.Compose([
        transforms.Resize(RESIZE_RESOLUTION),
        transforms.CenterCrop(CENTER_CROP),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STANDARD_DEVIATION
        )])

    img_transformed = transform(img)
    batched_tensor = torch.unsqueeze(img_transformed, 0)

    # print("Preprocessing took: %.4f s" % (time.time() - preprocessing_start_time))

    return batched_tensor


def get_classes(classes_json):
    """Returns a list of classes that should be used to classify the image"""
    print("Obtaining classes stored at: %s!" % classes_json)

    if not os.path.exists(os.path.dirname(classes_json)):
        print("Could not find the classes file: %s" % classes_json)
        exit(-1)

    with open(classes_json) as json_file:
        classes_list = json.load(json_file)

    return [classes_list[str(k)][1] for k in range(len(classes_list))]


def write_matched_classes_to_file(matched_classes, output_json):
    if not os.path.exists(os.path.dirname(output_json)):
        try:
            os.makedirs(os.path.dirname(output_json))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    print("Writing result to output file: %s" % output_json)

    with open(output_json, 'w') as classes_file:
        classes_file.write(json.dumps(matched_classes, indent=4))


def execute_classification(classes, batched_tensor, nr_output_classes):
    """
    Does an inference on a pre-trained ResNet101 model that was trained on the ImageNet dataset
    :param classes: The ImageNet classes that will be used to label the image
    :param batched_tensor: Image represented as a tensor
    :param nr_output_classes: The number of matched classes that should be returned
    :return: A dictionary containing the classifier as a key and the match percentage as a value
    """
    # print("Executing image classification")
    start_time_classify = time.time()

    classifier = models.inception_v3(pretrained=True)
    classifier.eval()
    out = classifier(batched_tensor)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    _, indices = torch.sort(out, descending=True)
    matched_classes = {classes[idx]: percentage[idx].item() for idx in islice(indices[0], nr_output_classes)}

    # print("Classification took: %.4f s" % (time.time() - start_time_classify))

    return matched_classes


def classify():
    start_time = time.time()

    args = parse_arguments()
    image_tensor = preprocess_image(args.filename)
    classes = get_classes(CLASSES_LOCATION)
    matched_classes = execute_classification(classes, image_tensor, args.count_classes)
    write_matched_classes_to_file(matched_classes, OUTPUT_FILE)

    print("Total execution time: %.4f s" % (time.time() - start_time))


if __name__ == '__main__':
    classify()





