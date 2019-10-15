import requests
import io

from torchvision import transforms
from PIL import Image


class ImageDownloader:
    transform_function = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])

    @staticmethod
    def transform_image(image_bytes):
        image = Image.open(image_bytes)
        transformed_image = ImageDownloader.transform_function(image)
        return transformed_image

    @staticmethod
    def get_image_from_link(download_link):
        response = requests.get(download_link)

        if response.status_code != 200:
            return None

        image_bytes = io.BytesIO(response.content)

        try:
            resized_image = ImageDownloader.transform_image(image_bytes)
        except Exception as e:
            return None

        return resized_image

