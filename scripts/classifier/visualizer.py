#!/usr/bin/env python

import image_classifier
import argparse
import os
import cv2
import time
import subprocess as sp


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

    return parser.parse_args()


def classify_photos(dataset):

    for file in os.listdir(dataset):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            file_path = os.path.join(dataset, filename)

            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            cv2.imshow('image', img)

            img_tensor = image_classifier.preprocess_image(file_path)
            img_classes = image_classifier.get_classes("./imagenet_classes.json")
            matched_classes = image_classifier.execute_classification(img_classes, img_tensor, 5)
            print(matched_classes)

            cv2.waitKey(0)
            cv2.destroyAllWindows()


def benchmark(dataset):

    nr_photos = 0
    total_preprocessing_time = 0
    total_classification_time = 0
    img_classes = image_classifier.get_classes("./imagenet_classes.json")

    for file in os.listdir(dataset):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            file_path = os.path.join(dataset, filename)
            nr_photos = nr_photos + 1

            preprocessing_start = time.time()
            img_tensor = image_classifier.preprocess_image(file_path)
            preprocessing_end = time.time()

            total_preprocessing_time = total_preprocessing_time + (preprocessing_end - preprocessing_start)

            classification_start = time.time()
            matched_labels = image_classifier.execute_classification(img_classes, img_tensor, 5)
            classification_end = time.time()

            total_classification_time = total_classification_time + (classification_end - classification_start)

            tmp = sp.call('clear', shell=True)
            print('{:<25s}{:5>d}'.format("Nr Photos:", nr_photos))
            print('{:<25s}{:5>.4f}'.format("Avg preprocessing time:", (total_preprocessing_time / nr_photos)))
            print('{:<25s}{:5>.4f}'.format("Avg classification time:", (total_classification_time / nr_photos)))


def visualize_photos():
    args = parse_arguments()
    # classify_photos(args.dataset)
    benchmark(args.dataset)

if __name__ == '__main__':
    visualize_photos()
