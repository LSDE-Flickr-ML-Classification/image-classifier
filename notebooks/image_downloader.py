import requests
import io

from torchvision import transforms
from PIL import Image
from multiprocessing.pool import ThreadPool


class ImageDownloader:
    transform_function = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    @staticmethod
    def get_image_as_normalized_tensor(image_bytes):
        image = Image.open(image_bytes)
        transformed_image = ImageDownloader.transform_function(image)
        return transformed_image

    @staticmethod
    def download_and_preprocess_image(flickr_data_row):
        link = flickr_data_row.photo_video_download_url
        response = requests.get(link)

        if response.status_code != 200:
            return None, None, None

        image_bytes = io.BytesIO(response.content)

        try:
            image_as_tensor = ImageDownloader.get_image_as_normalized_tensor(image_bytes)
        except Exception as e:
            # TODO: Handle the case when the image is grayscale
            return None, None, None

        return flickr_data_row.id, flickr_data_row.photo_video_download_url, image_as_tensor

    @staticmethod
    def download_images(links, threads_count):
        print("Downloading %d possible images on %d threads..." % (len(links), threads_count))

        tensor_images_it = ThreadPool(threads_count).imap_unordered(ImageDownloader.download_and_preprocess_image, links)
        tensor_images = []

        for tensor_image in tensor_images_it:
            if tensor_image is None:
                continue
            tensor_images.append(tensor_image)

        print("Succefuly Downloaded %d images" % len(tensor_images))
        return tensor_images
