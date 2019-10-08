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
    def download_images(links, threads):
        print("Downloading %d possible images..." % len(links))
        tensor_images = []
        for i in range(0, len(links)):
            row = links[i]
            response = requests.get(row.photo_video_download_url)
            if response.status_code != 200:
                continue

            image_bytes = io.BytesIO(response.content)

            try:
                image_as_tensor = ImageDownloader.get_image_as_normalized_tensor(image_bytes)
            except Exception as e:
                # TODO: Handle the case when the image is grayscale
                continue

            entry = (row.id, row.photo_video_download_url, image_as_tensor)
            tensor_images.append(entry)

        print("Succefuly Downloaded %d images" % len(tensor_images))
        return tensor_images
