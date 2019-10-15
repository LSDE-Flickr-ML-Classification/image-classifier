import requests
import io

from torchvision import transforms
from PIL import Image


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
    def get_normalized_photo(download_link):
        response = requests.get(download_link)

        if response.status_code != 200:
            return None

        image_bytes = io.BytesIO(response.content)

        try:
            image_as_tensor = ImageDownloader.get_image_as_normalized_tensor(image_bytes)
        except Exception as e:
            # TODO: Handle the case when the image is grayscale
            return None

        return image_as_tensor

